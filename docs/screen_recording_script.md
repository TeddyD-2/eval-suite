# Screen recording script — v0 demo

**Total length target: ~3:15 (195s).** Tighter is better; reviewers won't watch a 5-minute video unless promised. The extra 15s vs. the arms-only cut is the Go1 (quadruped) beat in the notebook walkthrough.

**Recording setup:**
- 1920×1080 capture, 30fps, mono mic.
- Use OBS or QuickTime; export as h.264 mp4 ≤ 50MB.
- Two windows visible: terminal (left, 1100px) + browser (right, 800px).
- Terminal font ≥ 16pt so text is readable in a thumbnail.
- Pre-stage: terminal cd'd to a fresh tmp dir; browser warmed up on `https://github.com/TeddyD-2/eval-suite`.

**Output:** `docs/screen_recording.mp4`. Linked from the README.

---

## Beat sheet

### 0:00 – 0:10 — Title card

**On screen:** Static slide with `eval-suite — v0 shipped` and the one-line idea.

**Narration:**
> Eval-suite. A reporting contract for robot foundation models — stop reporting a single success rate, report a per-condition profile, and ship a manifest so the numbers reproduce. V-zero ships the contract on two arms and a legged quadruped, with a signed-submission portal.

---

### 0:10 – 0:25 — Github page

**On screen:** Browser on `https://github.com/TeddyD-2/eval-suite`. Cursor on the README; scroll once past the badges to the "One-command reproduce" header.

**Narration:**
> Here's the repo. CI green. README has the one-command reproduce. The plan, the extension document, and the v-zero-point-one scoping all live alongside the code.

---

### 0:25 – 0:55 — Clone + build

**On screen:** Terminal, side by side with browser. Type (don't paste, so the audience can read along):
```bash
git clone https://github.com/TeddyD-2/eval-suite.git
cd eval-suite
docker build -t evalsuite .
```

**Narration (while build scrolls):**
> Clone. Build the reproducer image. The Dockerfile pins the SimplerEnv commit and the ManiSkill2 commit, plus the dependency pins you actually need to get the install to work in twenty-twenty-six — those were not in the SimplerEnv README. Build takes about fifteen minutes on a fresh box; we'll skip ahead.

**Editor note:** Cut here. Splice in a 0.5s "•••" card showing build success. Resume at the next beat with the prompt already showing the image is built.

---

### 0:55 – 1:30 — Docker run sweep

**On screen:** Terminal. Show the result of `docker images | grep evalsuite` for a half-second. Then run:
```bash
docker run --gpus all --rm \
    -v $PWD/checkpoints:/work/checkpoints \
    -v $PWD/results:/work/results \
    evalsuite \
    python -m eval_suite.cli sweep \
      --model-family rt1 \
      --rt1-ckpt-path /work/checkpoints/rt_1_tf_trained_for_000400120 \
      --task google_robot_pick_coke_can \
      --trials 20 \
      --output-dir /work/results/demo \
      --videos-dir /work/results/demo/videos
```

**Narration (during model load):**
> Run one sweep — R-T-1 on Google Robot pick coke can, twenty trials per cell, full variant grid. Twenty-nine cells total — orientation, lighting, background, distractor, table texture, plus a five-cell paraphrase axis for language robustness. The model loads once and amortizes the J-I-T compile across all five-hundred-eighty trials. About thirty minutes on a three-thousand-ninety; we'll cut to the end.

**Editor note:** Cut at first "trial 1 success=True" line. Splice 0.5s "•••" card. Resume on the final "sweep done" line showing total wall time and `run_id`.

---

### 1:30 – 1:50 — Show the artifacts on disk

**On screen:** Terminal, single command:
```bash
ls results/demo/ && echo --- && head -3 results/demo/trials.csv && echo --- && jq -r '.run_id, .cells | length' results/demo/manifest.json
```

**Narration:**
> Three artifacts. Trials C-S-V — one row per rollout. Manifest J-S-O-N with the content-addressed run-I-D. And a videos directory with one h-two-six-four m-p-four per rollout.

---

### 1:50 – 2:50 — Notebook walkthrough

**On screen:** Click into the running JupyterLab tab (already opened earlier). Open `takehome/profile.ipynb`. Run All. Then scroll through cells deliberately, pausing on each plot for 5-8 seconds.

**Narration, beat by beat:**

> *(scrolling to per-cell bar plot, pause 6s)*
> The per-cell profile. Each bar is one cell, twenty trials, Wilson ninety-five percent C-I. The red bars are the worst axis — that's the headline ranking metric.

> *(scrolling to per-axis means plot, pause 7s)*
> Per-axis means. The shape is the comparison. If R-T-1's worst axis is lighting and Octo's worst axis is distractor, those are different shapes — and the single-aggregate-score view that papers cite would hide that.

> *(scrolling to WidowX section, pause 4s)*
> WidowX is the second embodiment — different action space, different camera setup, same adapter. Platform validation, not a sweep.

> *(scrolling to Unitree Go1 section, pause 12s)*
> Third embodiment: Unitree Go1, a twelve-D-O-F joint-space quadruped via MuJoCo Playground. Same `run_sweep`, same `Manifest`, same notebook — only a sibling Adapter because the action class differs from seven-D-O-F end-effector. Per-cell bars under a random locomotion policy; the headline is that *terrain* is the worst axis across all three task families — flat survives, rough breaks down. Camera is a null axis, which is the sanity check. The framework absorbed an action-class change in one new value type plus a sibling Adapter. That's the v-zero substrate proof for a legged embodiment. Wiring up a trained Go1 controller is named in the README as mechanical follow-up.

> *(scrolling to sim-real overlay, pause 7s)*
> Sim versus published real for Octo on Google Robot. This is the single tier-B calibration point v-zero ships. The framework for adding more — tier-A paired comparisons as customer deployment data accrues — is in section four of the extension document.

> *(scrolling to manifest verify cell, pause 5s)*
> Manifest verify returns true — the run-I-D is the SHA-two-fifty-six of the canonical manifest contents minus the run-I-D itself. If anyone tampers with the artifact, verify returns false. That's the reproducibility claim made enforceable.

---

### 3:05 – 3:15 — Closer

**On screen:** Cut back to the README, scrolled to the "what's actually in the repo right now" section. Hold for 8s on the in/out lists.

**Narration:**
> Honest about what v-zero doesn't do. Two arms, one quadruped, one task family with twenty-nine cells, one tier-B calibration data point. The Go1 sweep is a substrate proof under a random policy; a trained controller and enforced sandboxing are v-one. The contribution here is the reporting contract — not the specific dataset.

**Editor note:** Fade out. Total: 180s.

---

## Cuts to make in editing

- Build step: 0:25 → 0:55 with a "•••" splice. Don't show 15 minutes of pip output.
- Sweep run: 0:55 → 1:30 with a "•••" splice. Show the kick-off command and the final completion line; skip the middle.
- Notebook cells should be pre-executed so "Run All" finishes in a few seconds on screen, not minutes.

## What to do if a take goes wrong

- If `docker build` fails on first take, that's still a valid sequence to show — re-run with cached layers; build should be ~10s on second take. Mention it in narration: "second take, the build is cached."
- If a sweep cell fails (rare but possible — e.g. a video frame size mismatch), let it continue; the failure becomes one of the failure rollouts in the curation section.
- If the notebook errors on first render against a partial sweep, that's actually a feature — the analysis module tolerates partial CSVs. Don't paper over it.

## Checklist before recording

- [ ] CI is green on `main`.
- [ ] Docker image builds locally with no warnings beyond the absl/sapien noise.
- [ ] Notebook executes against a finished sweep with all 4 arm (model, task) combos **plus the Go1 sweep** (12-cell, N=10, under random-locomotion policy). All 16 cells render without errors and outputs are committed.
- [ ] Curated success/failure mp4s referenced in the closer are committed to the repo and playable inline (arm rollouts in `docs/curated_videos/`; Go1 mp4s live under `results/sweep_*/random_go1_unitree_go1_joystick/videos/` and are gitignored — pick one or two and copy to `docs/curated_videos/` if you want them in the recording).
- [ ] Terminal font size set ≥ 16pt.
- [ ] Mic check: no fan noise, no notifications.
- [ ] Browser zoom set so the GitHub page text matches the terminal text height.
