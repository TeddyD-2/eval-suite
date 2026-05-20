"""CLI entry: `python -m eval_suite.cli sweep ...`.

The CLI is plugin-aware. Tasks, Policies, and Adapters are resolved
by short name from the entry-points registry — including any
third-party packages installed in the current Python environment.

  python -m eval_suite.cli sweep \\
      --task my_pkg:household_kitchen \\
      --policy my_pkg:industrial_arm \\
      --adapter gym \\
      --trials 10 --output-dir results/run/

The first invocation = one (model, task) sweep. The shell loop in
`scripts/run_full_sweep.sh` calls this once per (model, embodiment)
combination so a crash in one doesn't kill the whole run, and so the
model load happens per-process (one TF/JAX VM per model — they don't
cohabit cleanly).

**Lazy-load contract:** `--help` must NOT import the plugin classes.
It calls `registry.list_*()` (cheap, metadata-only) to populate the
help text and validates `--task NAME` against the catalog at parse
time. The actual `.load()` only happens after we've decided to run the
sweep — `registry.get_*(name)` is called inside `_run_sweep`, not at
module top. This is so `python -m eval_suite.cli --help` doesn't pull
in TensorFlow or JAX.

**Plugin kwargs.** Policies that need constructor arguments (e.g.
SimplerEnvPolicy needs `family`, `policy_setup`, `ckpt_path` /
`model_type`) accept them via `--policy-arg key=value` (repeatable).
There is only one entry path — every Policy, in-tree and third-party,
goes through it. `scripts/run_full_sweep.sh` is the worked example.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any


def _kv_args(values: list[str] | None) -> dict[str, Any]:
    """Parse repeated `key=value` strings into a kwargs dict."""
    out: dict[str, Any] = {}
    for v in values or []:
        if "=" not in v:
            raise ValueError(f"--*-arg must be key=value, got {v!r}")
        k, sep, val = v.partition("=")
        out[k.strip()] = val
    return out


def _build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser using the cheap-list registry catalog.

    Reads `registry.list_tasks()` etc. for the `--task` / `--policy` /
    `--adapter` help text. None of those calls triggers a plugin import.
    """
    # Lazy local import so `import eval_suite.cli` (without invoking main)
    # doesn't drag in importlib.metadata until needed. Cheap regardless.
    from .registry import list_adapters, list_policies, list_tasks

    p = argparse.ArgumentParser(prog="eval_suite")
    sub = p.add_subparsers(dest="cmd", required=True)

    sweep_p = sub.add_parser("sweep", help="run one (model, task) sweep")

    # Build help strings that surface installed plugins. argparse `choices=...`
    # is intentionally NOT used here — that would require list-time validation
    # which conflicts with the qualified `pkg:name` form for disambiguation.
    # The CLI validates after parsing, when registry.get_*() resolves.
    tasks = [e.name for e in list_tasks()]
    policies = [e.name for e in list_policies()]
    adapters = [e.name for e in list_adapters()]

    sweep_p.add_argument("--task", required=True,
                         help=f"Task plugin name. Installed: {sorted(tasks)}. "
                              f"Use pkg:name to disambiguate.")
    sweep_p.add_argument("--task-arg", action="append", default=[],
                         help="Constructor kwarg for the Task, repeatable. Format: key=value.")

    sweep_p.add_argument("--policy", required=True,
                         help=f"Policy plugin name. Installed: {sorted(policies)}.")
    sweep_p.add_argument("--policy-arg", action="append", default=[],
                         help="Constructor kwarg for the Policy, repeatable.")

    sweep_p.add_argument("--adapter", default="gym",
                         help=f"Adapter plugin name. Installed: {sorted(adapters)}. "
                              "Default: gym.")
    sweep_p.add_argument("--adapter-arg", action="append", default=[],
                         help="Constructor kwarg for the Adapter, repeatable.")

    sweep_p.add_argument("--trials", type=int, required=True, help="N seeds per cell.")
    sweep_p.add_argument("--seed-base", type=int, default=0,
                         help="Seeds are [seed_base, seed_base+1, ...].")
    sweep_p.add_argument("--cells", default=None,
                         help="Comma-separated cell indices to sweep (default: all).")
    sweep_p.add_argument("--output-dir", required=True,
                         help="Where to write trials.csv + manifest.json + plugin_provenance.json.")
    sweep_p.add_argument("--calibration-tier", choices=["A", "B", "C"], default="C")
    sweep_p.add_argument("--calibration-source", default="")
    sweep_p.add_argument("--calibration-value", type=float, default=None)
    sweep_p.add_argument("--notes", default="")
    sweep_p.add_argument("--container-digest", default="")
    sweep_p.add_argument(
        "--videos-dir", default=None,
        help="If set, write one rollout-dir per trial (rollout.mp4 + trajectory.npz + metadata.json).",
    )

    # Discovery subcommand: list installed plugins without running anything.
    list_p = sub.add_parser("list", help="list installed plugins (no rollout)")
    list_p.add_argument("--group", choices=["tasks", "policies", "adapters", "all"], default="all")

    return p


def _run_list(args: argparse.Namespace) -> int:
    """Dump the installed-plugin catalog. Cheap; no plugin .load()."""
    from .registry import list_adapters, list_failed, list_policies, list_tasks
    if args.group in ("tasks", "all"):
        print("# Tasks")
        for e in list_tasks():
            print(f"  {e.name:40} ({e.package_name}@{e.package_version})  -> {e.entry_point_ref}")
    if args.group in ("policies", "all"):
        print("# Policies")
        for e in list_policies():
            print(f"  {e.name:40} ({e.package_name}@{e.package_version})  -> {e.entry_point_ref}")
    if args.group in ("adapters", "all"):
        print("# Adapters")
        for e in list_adapters():
            print(f"  {e.name:40} ({e.package_name}@{e.package_version})  -> {e.entry_point_ref}")
    failed = list_failed()
    if any(failed.values()):
        print("# Failed-to-enumerate plugins")
        for grp, items in failed.items():
            for name, err in items:
                print(f"  [{grp}] {name}: {err}")
    return 0


def _run_sweep(args: argparse.Namespace) -> int:
    """Resolve the plugins, instantiate them, run the sweep.

    This is the FIRST place plugin `.load()` is called. Imports happen
    only now, not at module-top, not on --help. That's the lazy-load
    contract: a user typing `python -m eval_suite.cli --help` doesn't
    pay the TensorFlow / JAX import cost.
    """
    from .manifest import CalibrationRef
    from .registry import get_adapter, get_policy, get_task
    from .sweep import run_sweep

    # 1. Task
    task_cls = get_task(args.task)
    task = task_cls(**_kv_args(args.task_arg))

    # 2. Policy — resolve via the entry-points registry and apply --policy-arg kwargs.
    policy_cls = get_policy(args.policy)
    policy_kwargs = _kv_args(args.policy_arg)
    policy = policy_cls(**policy_kwargs)

    # 3. Adapter
    adapter_cls = get_adapter(args.adapter)
    adapter_kwargs = _kv_args(args.adapter_arg)
    if args.videos_dir:
        adapter_kwargs.setdefault("videos_dir", args.videos_dir)
    adapter = adapter_cls(**adapter_kwargs)

    seeds = list(range(args.seed_base, args.seed_base + args.trials))
    cells = [int(c) for c in args.cells.split(",")] if args.cells else None
    calibration = CalibrationRef(
        tier=args.calibration_tier,
        real_perf_source=args.calibration_source,
        real_perf_value=args.calibration_value,
    )
    run_sweep(
        policy=policy, task=task, adapter=adapter,
        seeds=seeds, cells=cells, output_dir=args.output_dir,
        container_digest=args.container_digest,
        calibration=calibration, notes=args.notes,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    p = _build_parser()
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if args.cmd == "list":
        return _run_list(args)
    if args.cmd == "sweep":
        return _run_sweep(args)
    p.error(f"unknown cmd: {args.cmd}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
