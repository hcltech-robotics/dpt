---
name: dpt-view-planning
description: Select imaging viewpoints with DPT. Use for planning n C-arm views around a CT organ or region, retaining the region on the detector while choosing complementary directions, or choosing the next calibrated view using supplied pose uncertainty and acquisition costs.
---

# DPT view planning

Choose the mode from the question and available inputs.

| Question | Mode | Required inputs |
|---|---|---|
| “Which three views cover this organ?” or “Choose n viewpoints around this area.” | Finite multi-view plan | CT, explicit region, candidate angles, number of views, circular source/detector geometry |
| “Which image should we acquire next to reduce target-position uncertainty?” | Local next-view design | Prepared case with CT attenuation, known pose, target points, current pose precision, calibrated candidates and costs |

Run commands from a DPT source checkout with DPT installed in the active Python environment. For an uninstalled checkout, prefix commands with `PYTHONPATH=python`. Geometry-only planning needs the `imaging` dependencies; projection and next-view design also need the `gpu` dependencies and a CUDA device. Use `--help` to inspect the installed example's arguments.

## Plan several views of a region

The executable is [`examples/agentic_view_planning.py`](../../examples/agentic_view_planning.py). The agent supplies the problem and reads the numerical result; this executable makes no model calls.

1. Identify the CT NIfTI file or DICOM series directory. For a directory with multiple series, use the reported `--series-uid`. Values default to HU; use `--units mm^-1` for supplied calibrated attenuation.
2. Define the region explicitly: a sphere with `--roi-centre-mm X Y Z --roi-radius-mm R`, a deliberately centred sphere with `--roi-at-volume-centre --roi-radius-mm R`, or `--roi-mask organ.nii.gz`. A named organ needs a supplied mask or a specified region. Use `--mask-label N` to select a positive label; otherwise all positive mask voxels are included. The mask must share the CT's native physical grid.
3. Supply the feasible angular candidates and required number of views. The default seven angles are an example search range. Source and detector distances, detector size and pixel spacing are explicit numerical inputs.
4. Inspect the region containment and candidate set, then save a plan to a fresh directory. Add `--render` when the agent needs selected-view DPT images to assess the result.

```bash
python examples/agentic_view_planning.py inspect \
  --ct /data/ct.nii.gz \
  --roi-at-volume-centre --roi-radius-mm 35 \
  --angles-degrees -90 -60 -30 0 30 60 90 --views 3

python examples/agentic_view_planning.py plan \
  --ct /data/ct.nii.gz \
  --roi-mask /data/organ.nii.gz --mask-label 1 \
  --angles-degrees -90 -60 -30 0 30 60 90 --views 3 \
  --output /data/runs/view-plan-001 --render --device cuda:0
```

Both commands return JSON on stdout. `inspect` evaluates geometry without writing files or using CUDA. `plan` saves `plan.json` and the exact sampled region points in `roi_points_ras_mm.npy`. With `--render`, it also saves selected-view `optical_depth.npy` and PNGs with one recorded display window.

### Coordinates, geometry and pose

- Region centres and `--isocentre-mm` use native CT RAS millimetres. The default isocentre is the physical CT centre, including its oblique affine. DICOM input is converted from LPS to RAS by the volume reader.
- Candidate angle zero puts the source on negative Y. The source position is `isocentre + SOD * (sin(angle), -cos(angle), 0)`; positive angles rotate about positive Z. Use the stored source and detector coordinates when converting this convention to a specific C-arm's angle names.
- Example geometry defaults: `--sod-mm 353.72339 --sdd-mm 702 --detector-shape 256 256 --pixel-spacing-mm 1.2 1.2`. Shape is rows/columns; spacing is column/row millimetres. Replace these values with supplied geometry when available.
- `--pose-json` accepts a rigid 4×4 JSON array or the DPT object `{"rotation": [nine row-major entries], "translation_mm": [x, y, z]}`. A result containing this transform under `pose_object_to_world` is also accepted. It maps CT coordinates to world coordinates. Both region points and candidate panels receive the transform, and the same pose is used for CT projection: the plan remains patient-relative.

### Score and agent decisions

`roi_coverage` reports the number and fraction of sampled region points that lie between the source and detector and project within detector edges. A sphere uses deterministic Fibonacci surface samples. A mask uses evenly spaced ranks of foreground voxel centres. `--roi-samples` sets the sample count from 16 to 4096, default 256; masks with fewer voxels use all of them.

A candidate is eligible when its sampled fraction meets `--min-coverage`, default 1. Among eligible n-view subsets, the planner maximises `det(sum(I - d d^T))`, where `d` is the unit isocentre-to-source direction. This dimensionless directional-complementarity score prefers independent viewing directions. The report includes the summed matrix and its minimum eigenvalue. Scores equal to ten decimal places prefer higher minimum containment, then the smaller sum of absolute angles, then ascending angles. Selection order is ascending angle.

The search accepts at most 24 unique angles in `[-180, 180)` and evaluates at most `--max-combinations` subsets, capped at 100000. A single direction has zero determinant; use next-view design when the task is uncertainty reduction for one additional image.

Read `status`, `search`, every candidate's containment counts and `selection` before reporting a plan. `no_feasible_set` retains the candidate scores with a null selection. Inspect which regions fall outside the detector and adjust only the available geometry, angular choices or requested coverage threshold, then use a fresh output directory. Keep the requested constraint explicit. Stop when the available choices cannot meet it.

Report selected angles, sample counts/fractions, directional score and saved geometry. Coverage describes sampled geometric containment; dose, tissue visibility and target-position covariance require their own models. DPT projections provide the selected CT images. For HU input, rendering records the 60 keV water-scaled attenuation model.

## Choose one next view from a prepared case

Use [`dpt.examples.acquisition_design`](../../python/dpt/examples/acquisition_design.py) when the question supplies current pose uncertainty and asks which next acquisition best reduces target-position variance.

Read the [acquisition-design case schema](https://hcltech-robotics.github.io/dpt/applications/cases/#acquisition-design) and [driver outputs](https://hcltech-robotics.github.io/dpt/applications/drivers/#acquisition-design-results). Prepare the actual arrays and hashes required by that schema. The case carries positive-definite current precision, scaled pose coordinates, object-frame targets, candidate geometry, open-beam expectations, feasibility descriptions and an acquisition-cost budget. Existing measurements contribute to current precision once.

```bash
python -m dpt.examples.acquisition_design \
  --case /data/design/case.json \
  --output /data/runs/next-view-001 --device cuda:0
```

Read `design.json` for the selected candidate, current and candidate mean target-position variance in mm², costs, exclusions and conditioning. The calculation combines expected ideal-Poisson Fisher information with the supplied current precision. Report that score with its supplied uncertainty and cost assumptions. Preserve `candidate_information.npy`, `candidate_local_covariance.npy` and `current_local_covariance.npy` for subsequent analysis.

If a prepared uncertainty case is absent and the question is about angular coverage of an organ, use the finite multi-view mode above. If current precision or acquisition calibration is essential to the requested question, obtain those inputs before running next-view design.
