# Agent skills

DPT includes a navigator and a skill for each agentic imaging workflow. The agent selects a calculation, runs its commands, inspects numerical results and images, and decides what to do next.

Start with **`$dpt-navigator`** when the workflow is unclear.

| Ask the agent | Selected skill | Result |
|---|---|---|
| Recover the CT pose from two calibrated count images | [Pose recovery](../skills/dpt-pose-recovery.md) | Object-to-world transform, per-view residuals and solver termination |
| Demonstrate pose recovery using this NIfTI or DICOM CT | [Pose recovery](../skills/dpt-pose-recovery.md) | Simulated AP/lateral images, fitted pose and assessment |
| Match the pose in an ordinary radiograph, including another patient's image | [Radiograph matching](../skills/dpt-match-radiograph.md) | Best visited pose and camera, measured loss reduction and overlay |
| Choose a set of views covering an organ or region | [View planning](../skills/dpt-view-planning.md) | Selected angles, sampled region coverage and complementary directions |
| Choose the next view to reduce target-position uncertainty | [View planning](../skills/dpt-view-planning.md) | One calibrated candidate, local covariance and cost |
| Project a CT or compose differentiable imaging operations | [Rendering integration](../skills/integrate-rendering.md) | Projection arrays and the requested derivatives |

The [navigator](../skills/dpt-navigator.md) explains the routing boundaries and how to combine pose matching with view planning.

## Use the skills

Use a source checkout and install the imaging example dependencies:

```bash
python -m pip install '.[gpu,imaging,examples]'
```

Git checkouts contain discovery links in `.agents/skills/`. Both checkouts and source archives include an `AGENTS.md` entry point and the canonical instructions in `skills/<name>/SKILL.md`. Agents that accept instruction files directly can open those files and follow their commands. The manual renders those same files as the skill pages linked above.

Examples of requests:

> Use `$dpt-navigator` with `scan.nii.gz` and `reference.png`. The radiograph is from another patient and has approximate geometry. Find a useful pose match and report the loss reduction.

> Use `$dpt-pose-recovery` to inspect `case.json`, fit its two calibrated views within 120 iterations and 800 solver evaluations, and report each view's residual.

> Use `$dpt-view-planning` to choose three views of the region in `organ-mask.nii.gz` from my CT. Evaluate the supplied candidate angles and show the chosen projections.

Each numerical run writes to a fresh output directory. The skills identify the result files and the evidence to inspect before continuing.

## Plan a set of views

For an explicitly requested central region:

```bash
python examples/agentic_view_planning.py inspect \
  --ct scan.nii.gz --roi-at-volume-centre --roi-radius-mm 35 \
  --angles-degrees -90 -60 -30 0 30 60 90 --views 3

python examples/agentic_view_planning.py plan \
  --ct scan.nii.gz --roi-at-volume-centre --roi-radius-mm 35 \
  --angles-degrees -90 -60 -30 0 30 60 90 --views 3 \
  --output outputs/view-plan-001 --render
```

For an organ segmentation, replace the sphere options with `--roi-mask organ-mask.nii.gz`. Supply the acquisition's source distance, detector distance and detector sampling when known. `plan.json` records the resolved geometry, sampled region, candidate scores and selected subset. Planning uses geometric calculations on the CPU; `--render` projects the selected views with DPT on CUDA.

The [view-planning skill](../skills/dpt-view-planning.md) also covers the prepared-case workflow for selecting the next view by expected Poisson information and target-position covariance.
