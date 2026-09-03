---
name: dpt-match-radiograph
description: Match a NIfTI or DICOM CT to the pose of an ordinary radiograph using DPT with approximate camera geometry, including references from a different patient. Search candidate orientations and reduce structural image loss; use dpt-pose-recovery for two calibrated primary-count images.
---

# DPT radiograph pose matching

Run commands from the DPT checkout root in an environment with `.[gpu,imaging,examples]` installed. Fitting requires CUDA. Use `examples/agentic_radiograph.py` to search camera and pose hypotheses, retain the lowest-loss visited pose, and inspect the resulting projection.

## Inspect the CT, image and geometry

The CT is a `.nii` or `.nii.gz` file, or a DICOM series directory, interpreted as HU. The reference can be PNG, JPEG, TIFF, a two-dimensional NumPy array or a single-frame DICOM image. It may depict another patient; preserve that provenance and judge the fit by pose plausibility and measured loss reduction.

```bash
python examples/agentic_radiograph.py inspect scan.nii.gz radiograph.png
```

For a DICOM CT directory with several series, select the intended CT using the loader's listed UIDs and add `--series-uid UID`. Read `status`, `ct.grid`, `ct.input_metadata`, `radiograph`, `camera` and `camera_hypotheses` in the JSON response. Confirm the intended anatomy, series, image orientation and physical volume extent before fitting.

The camera is centred on the CT. With no supplied dimensions, source/detector distances and coverage are derived from the volume extent. Supply available estimates consistently to both `inspect` and `fit`: `--sod-mm` is source-to-CT-centre distance, `--sdd-mm` is source-to-detector distance, and `--pixel-spacing-mm` is square detector pitch in the **original** radiograph. Unknown pitch triggers six field-of-view hypotheses. Supplying pitch fixes the field of view.

`--max-size` defaults to 192 pixels along the longest image edge, preserving aspect ratio; `--max-size 0` keeps native resolution. Detector pitch is adjusted for resizing. The native CT grid remains unchanged.

## Set and run the search budget

Use the user's limits. A representative run is:

```bash
python examples/agentic_radiograph.py fit scan.nii.gz radiograph.png \
  --output outputs/radiograph-fit-001 --device cuda:0 \
  --starts 3 --iterations 80 --angular-step-deg 45
```

Choose a fresh output directory for each run. Add the same CT selection, image-size and camera options used during inspection.

`--iterations` applies to **each** refined start. All seed hypotheses are scored first; `--starts` controls how many are refined. With angular step `A`, camera count `C`, starts `S` and iterations `I`, the limits are:

```text
seed projections       = (360 / A) * C
total projections      = seed projections + S * (I + 1)
backward evaluations   = S * I
```

At the settings above, unknown pitch gives 48 seeds, at most 291 projections and 240 backward evaluations. Fixed pitch gives eight seeds and at most 251 projections. `A` must divide 360 and lie between 15 and 180 degrees; `S` must fit the candidate count. Use `--translation-bound-mm` and `--rotation-bound-deg` when the task supplies search constraints; these bound local tangent components around each seed. Each call performs a fresh search.

## Inspect evidence and decide

The objective is `1 - abs(normalised cross-correlation)` against projected optical depth. Both brightness polarities are tested. Read the generated files:

| File | Decision evidence |
|---|---|
| `result.json` | `outcome`, `initial_loss`, `best_loss`, `loss_reduction`, `refinement_loss_reduction`, `score`, `signed_ncc`, `polarity`, selected pose and camera, evaluation counts and budget |
| `candidates.json` | Initial camera/orientation scores and refined candidates |
| `trajectory.json` | Every evaluated pose and score, including the selected evaluation |
| `settings.json` | Camera hypotheses, parameter bounds and numerical settings |
| `comparison.png`, `overlay.png` | Reference versus selected projection; overlay uses reference in magenta and projection in green |
| `optical_depth.npy` | Computed projection at the selected pose |

`initial_loss` uses the first seed; `refinement_loss_reduction` measures improvement over the best seed. Inspect both to distinguish candidate selection from local refinement. `projection.png` and the comparison images apply a fitted display gain/offset; use the stored score and optical depth for quantitative assessment.

Report the selected `pose_object_to_world` or `matrix_object_to_world` with its `camera`, `coordinate_frame: RAS`, `length_units: mm`, initial/best loss and actual evaluation counts. The result identifies the best pose visited under the chosen camera model. For a different patient's reference, describe a pose match and report the remaining structural mismatch.

- If `outcome` is `improved` and the image alignment is useful for the requested task, return the transform and comparison/trajectory paths.
- If improvement is small or several candidates look similarly plausible, inspect anatomy, field of view and orientation before choosing another run. Explain the remaining ambiguity using the saved candidates and overlay.
- If `outcome` is `no_improvement`, report the retained pose and measured result. Choose another search only when a concrete input, camera assumption or budget change addresses the evidence and remains within the user's scope.

Use `$dpt-pose-recovery` for two calibrated count images or a CT-only simulated recovery example. For calibrated transmission/optical-depth images, see `examples/fit_pose.py` and `docs/pytorch.md`. Further details are in `docs/applications/agentic-radiograph.md`.

Representative request: “Use `$dpt-match-radiograph` to make this CT match the projection pose in this radiograph from another patient. Search three starts for 80 iterations each, then show the best overlay and report the loss reduction and assumed camera geometry.”
