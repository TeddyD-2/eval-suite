"""Submission portal.

FastAPI app exposing:

  Submission API (JSON):
    POST   /submit                      — accept a signed manifest
    GET    /submissions                 — list every accepted/rejected
                                          (optional ?run_id=X filter)
    GET    /submissions/{run_id}        — full manifest + portal metadata
    GET    /healthz                     — 200 OK

  Plugin-registry API (JSON):
    GET    /registry/tasks              — installed Task plugins
    GET    /registry/policies           — installed Policy plugins
    GET    /registry/adapters           — installed Adapter plugins
    GET    /registry/failed             — plugins that failed to enumerate
    GET    /ledger                      — append-only submission ledger

  Browsable HTML UI (Courier New, Jinja2 templates, no JS):
    GET    /                            — 307 redirect to /ui/
    GET    /ui/                         — landing page
    GET    /ui/submissions              — filter form + paginated table
    GET    /ui/submissions/{run_id}     — full per-submission detail
    GET    /ui/registry                 — Tasks / Policies / Adapters
    GET    /ui/compare?a=&b=            — two-run side-by-side compare
    GET    /ui/ledger                   — append-only log as HTML
    GET    /ui/about                    — trust model + portal info
    GET    /static/style.css            — single CSS asset

Allow-list lives at $ALLOWED_KEYS_PATH (default `./allowed_keys.json`)
— JSON {hex_public_key: identity_string}. v0 treats this as a
manually-curated list; v1.0 moves to Sigstore-style keyless flow.

Run locally:
  uvicorn eval_suite.portal.app:app --reload --port 8000
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import urlencode

from fastapi import Body, FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader, select_autoescape

from ..contracts import CONTRACT_VERSION
from ..manifest import Manifest
from .storage import SubmissionStore, filter_submissions

_PORTAL_DIR = Path(__file__).resolve().parent
_TEMPLATES_DIR = _PORTAL_DIR / "templates"
_STATIC_DIR = _PORTAL_DIR / "static"

_jinja = Environment(
    loader=FileSystemLoader(_TEMPLATES_DIR),
    autoescape=select_autoescape(["html"]),
)


# ---- helpers shared across handlers ------------------------------------


def _load_allowed_keys(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    return {k.lower(): str(v) for k, v in raw.items()}


def _make_store() -> SubmissionStore:
    root = os.environ.get("EVAL_SUITE_SUBMISSIONS_DIR", "./submissions")
    return SubmissionStore(root)


def _allowed_keys_path() -> Path:
    return Path(os.environ.get("EVAL_SUITE_ALLOWED_KEYS", "./allowed_keys.json"))


def _registry_snapshot() -> dict[str, list[dict[str, str]]]:
    from ..registry import list_adapters, list_failed, list_policies, list_tasks

    def to_dicts(entries: list) -> list[dict[str, str]]:  # type: ignore[type-arg]
        return [
            {
                "name": e.name,
                "package_name": e.package_name,
                "package_version": e.package_version,
                "entry_point_ref": e.entry_point_ref,
                "qualified_name": e.qualified_name,
            }
            for e in entries
        ]

    return {
        "tasks": to_dicts(list_tasks()),
        "policies": to_dicts(list_policies()),
        "adapters": to_dicts(list_adapters()),
        "failed": [
            {"group": g.replace("eval_suite.", ""), "name": n, "error": err}
            for g, items in list_failed().items()
            for n, err in items
        ],
    }


def _render(template: str, **ctx: Any) -> HTMLResponse:
    """Render a Jinja2 template with the shared context (active page,
    contract_version) merged in."""
    ctx.setdefault("contract_version", CONTRACT_VERSION)
    html = _jinja.get_template(template).render(**ctx)
    return HTMLResponse(html)


def create_app(store: SubmissionStore | None = None) -> FastAPI:
    store = store or _make_store()
    app = FastAPI(title="eval-suite submission portal", version="0.8.0")

    app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

    # ----- JSON API ----------------------------------------------------

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/submit")
    def submit(payload: Annotated[dict[str, Any], Body()]) -> JSONResponse:
        manifest_json = payload.get("manifest")
        if not isinstance(manifest_json, str):
            raise HTTPException(status_code=400, detail="missing 'manifest' string in body")

        try:
            manifest = Manifest.from_json(manifest_json)
        except Exception as e:
            store.reject(manifest_json, reason=f"malformed manifest: {e}", attempted_identity=None)
            raise HTTPException(status_code=400, detail=f"malformed manifest: {e}") from e

        if not manifest.verify():
            store.reject(manifest_json, reason="manifest verify() failed",
                         attempted_identity=manifest.submitter_identity)
            raise HTTPException(status_code=400, detail="manifest verify() failed (hash or signature mismatch)")

        allowed = _load_allowed_keys(_allowed_keys_path())
        if allowed:
            if not manifest.submitter_signature or not manifest.submitter_public_key:
                store.reject(manifest_json, reason="signature required but absent",
                             attempted_identity=manifest.submitter_identity)
                raise HTTPException(status_code=403, detail="signature required (this portal enforces allow-list)")
            pk_lower = manifest.submitter_public_key.lower()
            if pk_lower not in allowed:
                store.reject(manifest_json, reason=f"public key {pk_lower[:16]}… not on allow-list",
                             attempted_identity=manifest.submitter_identity)
                raise HTTPException(status_code=403, detail="public key not on allow-list")
            object.__setattr__(manifest, "submitter_identity", allowed[pk_lower])

        sub = store.accept(manifest)
        corroborators = store.list_for_run_id(sub.run_id)
        n_other = sum(1 for s in corroborators if s.submitter_pk != sub.submitter_pk)
        return JSONResponse(
            content={
                "accepted": True,
                "run_id": sub.run_id,
                "embodiment": sub.embodiment,
                "task_name": sub.task_name,
                "submitter_identity": sub.submitter_identity,
                "ingest_ts": sub.ingest_ts,
                "corroborating_submitters": n_other,
            },
            status_code=201,
        )

    @app.get("/submissions")
    def list_subs(run_id: Annotated[str | None, Query()] = None) -> JSONResponse:
        all_subs = store.list_submissions()
        if run_id is not None:
            all_subs = [s for s in all_subs if s.run_id == run_id]
        return JSONResponse(content={
            "submissions": [asdict(s) for s in all_subs],
            "corroborating_submitters": len({s.submitter_pk for s in all_subs}) if run_id else None,
        })

    @app.get("/submissions/{run_id}")
    def get_sub(run_id: str) -> JSONResponse:
        result = store.get(run_id)
        if result is None:
            raise HTTPException(status_code=404, detail=f"no submission with run_id {run_id}")
        sub, manifest = result
        return JSONResponse(content={
            "submission": asdict(sub),
            "manifest": json.loads(manifest.to_json()),
        })

    @app.get("/registry/tasks")
    def registry_tasks() -> JSONResponse:
        return JSONResponse(content={"tasks": _registry_snapshot()["tasks"]})

    @app.get("/registry/policies")
    def registry_policies() -> JSONResponse:
        return JSONResponse(content={"policies": _registry_snapshot()["policies"]})

    @app.get("/registry/adapters")
    def registry_adapters() -> JSONResponse:
        return JSONResponse(content={"adapters": _registry_snapshot()["adapters"]})

    @app.get("/registry/failed")
    def registry_failed() -> JSONResponse:
        return JSONResponse(content={"failed": _registry_snapshot()["failed"]})

    @app.get("/ledger")
    def ledger() -> PlainTextResponse:
        path = store.ledger_path
        if not path.exists():
            return PlainTextResponse("")
        return PlainTextResponse(path.read_text(), media_type="application/x-ndjson")

    # ----- HTML UI ------------------------------------------------------

    @app.get("/", include_in_schema=False)
    def root_redirect() -> RedirectResponse:
        return RedirectResponse(url="/ui/", status_code=307)

    @app.get("/ui/", response_class=HTMLResponse)
    def ui_home() -> HTMLResponse:
        snap = _registry_snapshot()
        all_subs = store.list_submissions()
        return _render(
            "index.html",
            active="home",
            counts={
                "tasks": len(snap["tasks"]),
                "policies": len(snap["policies"]),
                "adapters": len(snap["adapters"]),
                "failed": len(snap["failed"]),
            },
            recent=all_subs[:10],
        )

    @app.get("/ui/submissions", response_class=HTMLResponse)
    def ui_submissions_list(
        model: str = "",
        task: str = "",
        embodiment: str = "",
        submitter: str = "",
        accepted: str = "any",
        page: int = 1,
    ) -> HTMLResponse:
        accepted_bool: bool | None = (
            True if accepted == "yes" else False if accepted == "no" else None
        )
        all_subs = store.list_submissions()
        filtered = filter_submissions(
            all_subs,
            model=model or None,
            task=task or None,
            embodiment=embodiment or None,
            submitter=submitter or None,
            accepted=accepted_bool,
        )
        per_page = 50
        total = len(filtered)
        total_pages = max(1, math.ceil(total / per_page))
        page = max(1, min(page, total_pages))
        start = (page - 1) * per_page
        end = min(start + per_page, total)
        page_rows = filtered[start:end]

        # Annotate corroborator counts cheaply on the in-memory list.
        corroborator_counts: dict[str, int] = {}
        for s in all_subs:
            corroborator_counts[s.run_id] = corroborator_counts.get(s.run_id, 0) + 1
        rows_with_counts: list[dict[str, Any]] = []
        for s in page_rows:
            d = asdict(s)
            d["corroborators"] = corroborator_counts.get(s.run_id, 1)
            rows_with_counts.append(d)

        filters = {"model": model, "task": task, "embodiment": embodiment,
                   "submitter": submitter, "accepted": accepted}
        has_filters = any([model, task, embodiment, submitter, accepted != "any"])

        def querystring_for_page(p: int) -> str:
            qs = {**{k: v for k, v in filters.items() if v not in ("", "any")}, "page": p}
            return urlencode(qs)

        return _render(
            "submissions_list.html",
            active="submissions",
            rows=rows_with_counts,
            filters=filters,
            has_filters=has_filters,
            page=page,
            total_pages=total_pages,
            total=total,
            start=start + 1 if total else 0,
            end=end,
            querystring_for_page=querystring_for_page,
        )

    @app.get("/ui/submissions/{run_id}", response_class=HTMLResponse)
    def ui_submission_detail(run_id: str) -> HTMLResponse:
        result = store.get(run_id)
        if result is None:
            # Minimal 404 HTML — not a full base-template page; reviewers
            # rarely land here and a short body is friendlier than a long one.
            return HTMLResponse(
                f"<html><body><h1>not found</h1>"
                f"<p>no submission with run_id <code>{run_id}</code> in this portal.</p>"
                f'<p><a href="/ui/submissions">&larr; back to submissions</a></p></body></html>',
                status_code=404,
            )
        sub, manifest = result
        from ..analysis import canonical_profile_from_manifest
        from ..statistics import per_axis_means

        profile = canonical_profile_from_manifest(manifest)

        # per-axis means need CellResult objects; we already synthesize them
        # inside canonical_profile_from_manifest. Recompute here cheaply
        # from the manifest's payload.
        from .._types import CellId, CellResult
        synthetic_cells = [
            CellResult(
                cell=CellId(embodiment=manifest.embodiment, task=manifest.task_name, axes=dict(c.axes)),
                n_trials=c.n_trials, successes=c.successes,
                wilson_ci_low=c.wilson_ci_low, wilson_ci_high=c.wilson_ci_high,
                per_seed_success=[],
            )
            for c in manifest.cells
        ]
        per_axis_raw = per_axis_means(synthetic_cells)
        # filter to multi-level axes for the per-axis section
        per_axis = {a: lvls for a, lvls in per_axis_raw.items() if len(lvls) > 1}

        corroborators = store.list_for_run_id(manifest.run_id)

        return _render(
            "submission_detail.html",
            active="submissions",
            submission=sub,
            manifest=manifest,
            run_id_short=manifest.run_id[:16],
            checkpoint_truncated=(manifest.model.checkpoint_sha256[:24] + "…")
                if manifest.model.checkpoint_sha256 else "—",
            contract_version_str=CONTRACT_VERSION,
            eval_suite_version_str="0.1.0",  # static; sidecar value would be more authoritative
            verify_ok=manifest.verify(),
            canonical_profile=profile,
            dims=["language", "visuals", "physics", "embodiment"],
            per_axis=per_axis,
            corroborators=corroborators,
        )

    @app.get("/ui/registry", response_class=HTMLResponse)
    def ui_registry() -> HTMLResponse:
        snap = _registry_snapshot()
        return _render(
            "registry.html",
            active="registry",
            **snap,
        )

    @app.get("/ui/compare", response_class=HTMLResponse)
    def ui_compare(
        request: Request,
        a: str = "",
        b: str = "",
    ) -> HTMLResponse:
        from ..analysis import canonical_profile_from_manifest

        all_subs = store.list_submissions()
        # Distinct accepted submissions for the picker dropdown.
        accepted = [s for s in all_subs if s.accepted]
        # Distinct by run_id (multiple submitters of the same run get one option).
        seen: set[str] = set()
        options: list[dict[str, str]] = []
        for s in accepted:
            if s.run_id in seen:
                continue
            seen.add(s.run_id)
            label = (
                f"{s.model_name or '—'} / {s.embodiment}/{s.task_name} "
                f"({s.run_id[:16]})"
            )
            options.append({"run_id": s.run_id, "label": label})

        def _build_side(run_id: str) -> dict[str, Any] | None:
            r = store.get(run_id)
            if r is None:
                return None
            _, manifest = r
            profile = canonical_profile_from_manifest(manifest)
            total_trials = sum(c.n_trials for c in manifest.cells)
            total_successes = sum(c.successes for c in manifest.cells)
            overall = total_successes / total_trials if total_trials else 0.0
            return {
                "run_id_short": manifest.run_id[:16],
                "manifest": manifest,
                "profile": profile,
                "total_trials": total_trials,
                "overall": overall,
            }

        a_side = _build_side(a) if a else None
        b_side = _build_side(b) if b else None
        have_both = a_side is not None and b_side is not None

        return _render(
            "compare.html",
            active="compare",
            options=options,
            have_both=have_both,
            same_run=have_both and a == b,
            a=a_side,
            b=b_side,
            a_id=a,
            b_id=b,
            dims=["language", "visuals", "physics", "embodiment"],
            share_url=str(request.url),
        )

    @app.get("/ui/ledger", response_class=HTMLResponse)
    def ui_ledger() -> HTMLResponse:
        path = store.ledger_path
        entries: list[dict[str, Any]] = []
        if path.exists():
            for line in path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        # Newest first
        entries.sort(key=lambda e: e.get("ts", 0), reverse=True)
        return _render("ledger.html", active="ledger", entries=entries)

    @app.get("/ui/about", response_class=HTMLResponse)
    def ui_about() -> HTMLResponse:
        return _render("about.html", active="about")

    return app


# Module-level app for `uvicorn eval_suite.portal.app:app ...`
app = create_app()
