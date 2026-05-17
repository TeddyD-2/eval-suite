# eval-suite-core

The eval-suite substrate: Protocol-based contracts (`Policy`, `Task`, `Adapter`, `Manifest`), content-addressed signed manifests, sweep driver, analysis, statistics, entry-points-based plugin registry, FastAPI submission portal, and the conformance kit plugin authors call from their own pytest.

This is the package third-party plugin authors depend on. The in-tree reference plugins (SimplerEnv tasks, Octo / RT-1 policies, the gym/MJX adapters) live in [`eval-suite-stdlib`](../eval-suite-stdlib/), structurally identical to a third-party plugin so the contract gets proven on the same path external code travels.

See the repo-root `README.md` for the user-facing vision and quick-start, and `takehome/EXTENSION.md` for the design doc.
