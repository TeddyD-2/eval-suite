"""Plugin registry.

**In plain words.** This is how new pieces — a task, a model, a
simulator bridge — get into the suite without anyone editing the
suite's own source code. A third-party package declares itself in its
`pyproject.toml` ("here is a task named X"); after `pip install`, this
file notices it and the CLI lists it alongside the built-in ones. The
suite is *extendable by installing pip packages*, not by forking the
repo, and that promise is implemented here.


Discovery of installed plugins via Python's standard `importlib.metadata`
entry-points. Three groups:

  - `eval_suite.tasks`     — Task implementations
  - `eval_suite.policies`  — Policy implementations
  - `eval_suite.adapters`  — Adapter implementations

A third-party package declares its entries in its own `pyproject.toml`:

    [project.entry-points."eval_suite.tasks"]
    my_kitchen_task = "my_pkg.kitchen:MyKitchenTask"

After `pip install my_pkg`, this registry sees that entry alongside
the in-tree ones; the CLI lets you `--task my_kitchen_task` without
modifying eval-suite source.

**Lazy by design:** `list_tasks()` / `list_policies()` / `list_adapters()`
return metadata only — they never call `.load()`. Loading happens
exclusively in `get_task(name)` / `get_policy(name)` / `get_adapter(name)`.
This is so `python -m eval_suite.cli --help` doesn't import TensorFlow,
JAX, or SimplerEnv just to print the option catalog.

**Error-tolerant:** if a plugin's module-level import fails (missing
optional dep, broken egg-info, etc.), the whole listing call doesn't
crash. The failing plugin lands in `list_failed()` with the exception
message attached.

**Strict-and-loud name resolution:** when two installed plugins register
the same short name, `get_*(name)` raises `EvalSuiteAmbiguousPluginError`
with a message listing the qualified forms. Permissive first-match
lookup would silently bind to whichever plugin loaded first — a
hard-to-debug failure six months from now.
"""

from __future__ import annotations

import importlib.metadata
from dataclasses import dataclass
from typing import Any, Literal

GroupName = Literal["eval_suite.tasks", "eval_suite.policies", "eval_suite.adapters"]


class EvalSuiteAmbiguousPluginError(Exception):
    """Two plugins claim the same short name; user must use the qualified form."""


class EvalSuitePluginNotFoundError(Exception):
    """No plugin is registered under the given (short or qualified) name."""


@dataclass(frozen=True)
class PluginEntry:
    """Metadata about one registered plugin entry.

    `name` is the short name (the key on the left of `=` in the
    pyproject.toml entry-point declaration). `package_name` is the
    pip distribution name; together with `name`, the qualified form
    is `f"{package_name}:{name}"` — used to disambiguate when two
    distributions register the same short name.
    """

    name: str
    package_name: str
    package_version: str
    entry_point_ref: str  # e.g. "eval_suite.tasks.simpler_env:GoogleRobotPickCokeCan"
    group: GroupName

    @property
    def qualified_name(self) -> str:
        return f"{self.package_name}:{self.name}"


# Cache of failed loads per group. Populated lazily by `_collect`.
_FAILED: dict[GroupName, list[tuple[str, str]]] = {
    "eval_suite.tasks": [],
    "eval_suite.policies": [],
    "eval_suite.adapters": [],
}


def _collect(group: GroupName) -> list[PluginEntry]:
    """Walk entry-points for a group, gathering metadata without loading.

    `importlib.metadata.entry_points` returns metadata objects that
    expose `.name`, `.value` (the "module:attr" ref), and `.dist`
    (the source distribution). We never call `.load()` here — that's
    the lazy contract.
    """
    out: list[PluginEntry] = []
    # Clear the failed list for this group; we rebuild every call so
    # a `pip install` between calls is picked up.
    _FAILED[group] = []
    eps = importlib.metadata.entry_points(group=group)
    for ep in eps:
        try:
            dist = ep.dist
            if dist is None:
                pkg_name = "unknown"
                pkg_version = "0.0.0"
            else:
                pkg_name = dist.metadata["Name"]
                pkg_version = dist.version
            out.append(PluginEntry(
                name=ep.name,
                package_name=pkg_name,
                package_version=pkg_version,
                entry_point_ref=ep.value,
                group=group,
            ))
        except Exception as e:
            _FAILED[group].append((ep.name, f"{type(e).__name__}: {e}"))
    return out


def list_tasks() -> list[PluginEntry]:
    return _collect("eval_suite.tasks")


def list_policies() -> list[PluginEntry]:
    return _collect("eval_suite.policies")


def list_adapters() -> list[PluginEntry]:
    return _collect("eval_suite.adapters")


def list_failed() -> dict[str, list[tuple[str, str]]]:
    """Per-group `(name, error_msg)` list of plugins that didn't enumerate cleanly.

    Note this is metadata-time failures (dist resolution, etc.), not
    runtime import failures — those surface in `get_*(name)` since
    that's where `.load()` happens.
    """
    # Trigger a refresh of each group so _FAILED is up-to-date.
    list_tasks()
    list_policies()
    list_adapters()
    return {k: list(v) for k, v in _FAILED.items()}


def _resolve(group: GroupName, name: str) -> PluginEntry:
    """Find a single PluginEntry matching `name` in `group`.

    `name` may be short ("mock") or qualified ("eval-suite:mock").
    Strict-and-loud: ambiguous short names raise; not found raises.
    """
    entries = _collect(group)
    # Qualified form: "package:short_name"
    if ":" in name:
        pkg, short = name.split(":", 1)
        for e in entries:
            if e.package_name == pkg and e.name == short:
                return e
        raise EvalSuitePluginNotFoundError(
            f"No plugin registered under {group} as {name!r}. "
            f"Available: {sorted(e.qualified_name for e in entries)}"
        )
    # Short form: must be uniquely-named.
    matches = [e for e in entries if e.name == name]
    if not matches:
        raise EvalSuitePluginNotFoundError(
            f"No plugin registered under {group} as {name!r}. "
            f"Available: {sorted(e.name for e in entries)}"
        )
    if len(matches) > 1:
        raise EvalSuiteAmbiguousPluginError(
            f"Multiple plugins claim the name {name!r} under {group}: "
            f"{sorted(e.qualified_name for e in matches)}. "
            f"Use the qualified form (e.g. {matches[0].qualified_name!r}) to disambiguate."
        )
    return matches[0]


def _load(entry: PluginEntry) -> type:
    """Resolve a PluginEntry to the actual class. Lazy — only call this
    when the user has decided to use this plugin (not on listing or --help)."""
    eps = importlib.metadata.entry_points(group=entry.group)
    for ep in eps:
        if ep.name == entry.name and (ep.dist is None or ep.dist.metadata["Name"] == entry.package_name):
            cls = ep.load()
            if not isinstance(cls, type):
                raise TypeError(
                    f"Entry point {entry.qualified_name} resolved to {type(cls).__name__}, "
                    f"expected a class."
                )
            return cls
    raise EvalSuitePluginNotFoundError(
        f"Could not resolve entry-point {entry.qualified_name} at load time "
        f"(it appeared in listing but is gone now — was a package uninstalled?)."
    )


def get_task(name: str) -> type:
    """Resolve and import the Task class registered as `name`.

    Triggers `.load()` and therefore module-level import side effects
    of the plugin (e.g. importing SimplerEnv). Don't call on `--help`.
    """
    return _load(_resolve("eval_suite.tasks", name))


def get_policy(name: str) -> type:
    return _load(_resolve("eval_suite.policies", name))


def get_adapter(name: str) -> type:
    return _load(_resolve("eval_suite.adapters", name))


def instantiate(group: GroupName, name: str, **kwargs: Any) -> Any:
    """Resolve + load + instantiate. Convenience wrapper used by the CLI."""
    if group == "eval_suite.tasks":
        cls = get_task(name)
    elif group == "eval_suite.policies":
        cls = get_policy(name)
    elif group == "eval_suite.adapters":
        cls = get_adapter(name)
    else:
        raise ValueError(f"unknown plugin group: {group}")
    return cls(**kwargs)
