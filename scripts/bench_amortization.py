"""Quick benchmark: does TF/JAX compile amortize across trials in one process?

Hypothesis: trial 1 dominated by compile (~30-40s), trials 2+ are short
(~5-15s steady state). If true, sweep CLI must keep the policy live for
all N trials per cell. If not, we need a different approach.

Usage:
  python scripts/bench_amortization.py --family rt1 \
      --ckpt /home/teddy/simpler-env/checkpoints/rt_1_tf_trained_for_000400120 \
      --trials 3 --cell 0
  python scripts/bench_amortization.py --family octo --model-type octo-base \
      --trials 3 --cell 0
"""

from __future__ import annotations

import argparse
import time

from eval_suite.adapters import GymAdapter
from eval_suite.policies.simpler_env import SimplerEnvPolicy
from eval_suite.tasks.simpler_env import GoogleRobotPickCokeCan


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--family", choices=["rt1", "octo"], required=True)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--model-type", default="octo-base")
    p.add_argument("--cell", type=int, default=0)
    p.add_argument("--trials", type=int, default=3)
    args = p.parse_args()

    task = GoogleRobotPickCokeCan()
    adapter = GymAdapter()

    print(f"[bench] family={args.family} cell={args.cell} ({task.cell_id(args.cell).axes})")

    t_load = time.monotonic()
    policy = SimplerEnvPolicy(
        family=args.family,
        policy_setup="google_robot",
        ckpt_path=args.ckpt,
        model_type=args.model_type,
    )
    load_s = time.monotonic() - t_load
    print(f"[bench] model load: {load_s:.1f}s")

    for trial_idx in range(args.trials):
        seed = trial_idx * 1000 + 7
        t0 = time.monotonic()
        result = adapter.rollout(policy, task, cell=args.cell, seed=seed)
        wall = time.monotonic() - t0
        print(
            f"[bench] trial {trial_idx + 1}: wall={wall:.1f}s  "
            f"adapter_reported={result.elapsed_wall_seconds:.1f}s  "
            f"steps={result.num_steps}  success={result.success}"
        )


if __name__ == "__main__":
    main()
