"""Objaverse-XL → MJCF ingest path.

The "factory engineer drops public objects into a sim scene" story:
pick an Objaverse-XL asset id (or a small set), fetch the .glb / .obj
via the `objaverse` Python client, convert to MJCF-friendly geometry
through `trimesh`, and compose it into the standard MJCF artifact
shape via the existing `compose_with_annotations` machinery — so
`ParametricSplatTask` runs against the result unchanged.

License gating: Objaverse contains a mix of CC-BY, CC-BY-NC, and
CC-0 assets. We default to a CC-BY + CC-0 allowlist (the
commercial-friendly subset). The allowlist is enforced at fetch time;
unlisted-license assets are refused with a clear error.
"""
