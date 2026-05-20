"""Objaverse-XL → MJCF ingest path.

**In plain words.** The Objaverse library is a giant collection of
public 3D objects (over 10 million of them). This ingest path lets
the suite fetch one of them by ID and drop it into a sim scene —
useful when a user wants to evaluate against a published asset
rather than a scan they captured themselves. License gating is
enforced *before* download, so a non-commercial-licensed asset
can't accidentally end up in a commercial run.


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
