---
name: dpt-navigator
description: "Choose the DPT skill for a CT imaging question: calibrated pose recovery, ordinary-radiograph pose matching, organ or region viewpoint planning, or differentiable rendering. Use when the task spans workflows or the required imaging workflow is unclear."
---

# Navigate DPT imaging

Match the requested result and available data to the table below. Open the selected skill before running its workflow. Commands run from a DPT source checkout with `.[gpu,imaging,examples]` installed; each specialist identifies the operations that require CUDA.

| Question or requested result | Inputs that decide the route | Skill |
|---|---|---|
| “Recover the CT pose from these two calibrated X-rays.” | Two primary-count images, detector calibration, open-beam counts and an initial object-to-world pose | [dpt-pose-recovery](../dpt-pose-recovery/SKILL.md) |
| “Show an agent recovering the pose of this CT.” | A NIfTI CT or DICOM series; reference projections will be simulated | [dpt-pose-recovery](../dpt-pose-recovery/SKILL.md), CT-only demonstration |
| “Make this CT look as though it was taken from the same pose as this radiograph.” | An ordinary radiograph with approximate or unknown geometry, including a different patient's image | [dpt-match-radiograph](../dpt-match-radiograph/SKILL.md) |
| “Choose three views that keep this organ in frame.” | CT, explicit organ mask or geometric region, candidate angles and image count | [dpt-view-planning](../dpt-view-planning/SKILL.md), geometric coverage mode |
| “Which next view will reduce uncertainty in the target position?” | Prepared calibrated candidates, current pose precision, target points, exposure and cost budget | [dpt-view-planning](../dpt-view-planning/SKILL.md), information mode |
| “Render a CT projection, differentiate it, or integrate spectral imaging into my code.” | Volume, acquisition geometry and the desired forward calculation or derivative | [integrate-rendering](../integrate-rendering/SKILL.md) |

## Resolve the boundary

Inspect supplied files and metadata before asking for information already present. When pose fitting is requested, determine whether the observations are calibrated numerical measurements or an ordinary image. Two files alone do not establish calibration. Route uncertain geometry to radiograph matching when that matches the user's intent; ask which image to fit if several ordinary radiographs are supplied to the single-image workflow.

For more than two calibrated count images, load `dpt-pose-recovery` and use its general registration route, retaining all supplied views. For calibrated transmission or optical-depth arrays, that skill points to `examples/fit_pose.py` and its explicit image-quantity setting.

For view planning, identify the region and objective. A named organ needs a supplied mask or a user-defined geometric region. Use a volume-centred sphere only when the user requests that region. Geometric coverage mode measures the fraction of sampled region points inside each detector and chooses complementary directions. Information mode selects one next view using local target-position covariance and an explicit cost budget. Choose the objective the user asked for.

If there is no CT or compatible volume, identify that missing input before starting a volume-based calculation. If the user asks for segmentation, hardware movement or dose planning, identify the additional operation and inputs required; these skills provide projection, registration and acquisition selection.

## Combine workflows

For “match this radiograph, then plan views”, first complete `dpt-match-radiograph`, retain the selected camera and object-to-world transform, then load `dpt-view-planning`. Pass the transform through the planner's pose input and keep the region in CT coordinates. Preserve the first run's files and give the next calculation a fresh output directory.

After choosing a route, state the selected skill and the deciding input in one sentence. Follow that skill's bounded calculation and result checks. Report the result the workflow actually measures: pose and residuals, radiograph loss reduction, geometric coverage and selected angles, or target-position uncertainty.

Representative request: “Use `$dpt-navigator` to choose the right workflow for my CT and radiograph. The radiograph is from another patient and has no camera calibration.”
