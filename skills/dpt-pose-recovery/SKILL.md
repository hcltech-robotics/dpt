---
name: dpt-pose-recovery
description: Recover a shared rigid CT pose from exactly two calibrated primary-count images with DPT, or run a pose-recovery demonstration by simulating two views of a supplied NIfTI or DICOM CT. Use dpt-match-radiograph for an ordinary radiograph with approximate geometry.
---

# DPT pose recovery

Run commands from the DPT checkout root in an environment with `.[gpu,imaging,examples]` installed. Simulation and fitting require CUDA. The agent chooses the calculation and budget, inspects the JSON results, then decides whether the evidence supports stopping.

## Select the inputs

- **CT only:** accept `.nii`, `.nii.gz` or a DICOM series directory. This workflow generates AP and lateral count images from the supplied CT. Record their simulated origin when reporting the result.
- **CT and exactly two calibrated count images:** use a registration case manifest. Read `docs/applications/cases.md` for the schema and `docs/applications/agentic-imaging.md` for the two-view contract.
- **More than two calibrated count images:** retain all views and use `python -m dpt.examples.registration`, described below.
- **Ordinary radiograph, approximate geometry or another patient's projection:** use `$dpt-match-radiograph`.

For a DICOM directory containing several series, inspect the loader's listed series and select the intended CT with `--series-uid UID`. Preserve native voxel spacing, origin and orientation; record source and usage terms when supplied. A prepared count case contains non-negative FP32 attenuation in `mm^-1`, FP32 integer counts in `[0, 2**24]`, binary FP32 masks, positive open-beam counts, detector geometry and input hashes. Excluded observations must be zero. Both detectors and the object-to-world anchor must use one physical coordinate frame.

## Prepare and inspect

For a CT-only demonstration, choose a new case directory:

```bash
python examples/agentic_imaging.py prepare-ct scan.nii.gz \
  --output outputs/pose-case-001 --device cuda:0
python examples/agentic_imaging.py inspect \
  --case outputs/pose-case-001/case.json
```

Replace `scan.nii.gz` with the DICOM directory and add `--series-uid UID` when needed. `prepare-ct` defaults to HU, 128-pixel square detectors and 768 samples per ray. `--source` and `--rights` preserve supplied provenance. `case.json` records the initial pose and calibration; `reference.json` holds the generating pose for later assessment.

For supplied measured data, run `inspect` directly on its case manifest. Require `status: inputs_valid` and check both view identifiers, shapes, included-pixel counts, chart and initial pose. Resolve input errors before fitting. Keep the supplied volume, masks, calibration and anchor unchanged during the fit. The chart is `object_to_world = anchor @ exp((scales * parameters)^)`, a local increment on the right. Scales are ordered translation x/y/z in millimetres, then rotation x/y/z in radians; solver parameters are dimensionless. Export the resolved object-to-world transform.

## Fit within the budget

Use the user's budget; the following is a bounded example:

```bash
python examples/agentic_imaging.py fit \
  --case outputs/pose-case-001/case.json \
  --output outputs/pose-fit-001 --device cuda:0 \
  --max-iterations 120 --max-evaluations 800
python examples/agentic_imaging.py assess --run outputs/pose-fit-001
```

Each fit requires a fresh output directory and starts from the case's anchor. `--max-evaluations` bounds joint solver objective evaluations; the tool then performs two additional per-view diagnostic evaluations. CT preparation has its own projection work. For a complete CT-only run, `run scan.nii.gz --output outputs/pose-run-001` performs preparation, inspection, fitting and assessment, and accepts the same two solver-budget flags.

Use a known generating or independent reference only after fitting:

```bash
python examples/agentic_imaging.py assess --run outputs/pose-fit-001 \
  --reference outputs/pose-case-001/reference.json
```

## Read the result and choose the next action

Read `summary.json` and `optimisation.json` from the fit directory. Report `status`, `termination`, `initial_loss`, `final_loss`, `gradient_infinity_norm`, `solver_evaluations` and `additional_final_view_evaluations`. Export `pose_object_to_world`, including rotation and translation in millimetres. Check every entry in `views`, particularly `half_deviance`, `residual_rms_counts` and `residual_max_abs_counts`, and inspect the saved initial/fitted predictions and residuals.

- If `status` is `stationary`, report that the fitted objective reached stationarity. Report geometric accuracy only when assessment supplies reference errors.
- If `status` is `unresolved`, use `termination` and the loss history to identify the next action. A budget termination may justify a larger budget within the user's scope; a calibration, image or mask issue needs correction first. A further fit starts from the unchanged case anchor and needs a new directory.
- If inputs fail validation, stop the numerical work and identify the failing input.

For more than two calibrated count images, put the solver budget in the manifest's `registration.policy` and run:

```bash
python -m dpt.examples.registration --case inputs/registration-case.json \
  --output outputs/registration-001 --device cuda:0
```

Read `optimisation.json` for the accepted transform, policy and solver status, and `fit-report.json` for all view diagnostics and evaluation counts. The general driver accepts a non-empty view list. See `docs/applications/drivers.md` for its output contract. Calibrated transmission or optical-depth images use `examples/fit_pose.py` with the matching `image_quantity`, as described in `docs/pytorch.md`.

Representative request: “Use `$dpt-pose-recovery` to inspect this two-view case, fit once within 120 iterations and 800 solver evaluations, and report the transform, each view's residual and the termination reason.”
