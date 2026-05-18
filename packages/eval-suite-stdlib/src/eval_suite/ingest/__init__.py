"""Asset-ingest plugins for the eval-suite stdlib.

Subpackages here turn third-party asset formats (Gaussian splats, USD scans,
URDFs, etc.) into MJCF scenes the existing Adapters can consume. They live
in `eval-suite-stdlib`, not in `eval-suite-core`, so the core contract has
zero dependency on heavy/GPU tooling. Users install with extras like
`pip install eval-suite-stdlib[splat]`.

Generic utilities shared across ingest paths (mesh decimation, byte-stable
XML serialization, asset hashing) live in `_mesh_utils.py`. The
`namaqualand_scan` USD ingestion path under `assets/` keeps its own
historical copy of these helpers — v1 intentionally does NOT reach in to
touch that path because the existing Namaqualand contract tests are the
proof-of-concept that the substrate accepts a new asset class without
contract changes; refactoring it for code-sharing has zero v1 demo benefit
and adds regression risk. v1.5 reunifies if a real second use appears.
"""
