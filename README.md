# DPT: Differentiable Photon Transport

**Chris von Csefalvay**

[Documentation](https://hcltech-robotics.github.io/dpt/) · [Performance](https://hcltech-robotics.github.io/dpt/performance/) · [Agentic imaging](https://hcltech-robotics.github.io/dpt/applications/agentic-imaging/) · [Citation](#citation)

DPT turns a CT volume and physical acquisition geometry into differentiable X-ray projections on CUDA. It provides PyTorch volume and pose derivatives, spectral image formation, detector operations and stochastic photon transport.

## Benchmark timings

Measured on an **NVIDIA RTX PRO 6000 Blackwell Workstation Edition** with a **256³ chest CT**, **360 views**, **512² detectors** and **1024 clipped midpoint samples per ray** for DPT.

![FP32 and FP64 median wall times for forward projection, volume gradients, pose gradients and a full iteration](docs/assets/figures/fullscan/fullscan-latency.png)

**Forward and gradient timings.** Completed resident PyTorch calls, with medians from 30 samples per operation and implementation. A full iteration includes the forward call and both volume and pose gradients.

![Completed forward latency versus relative image error for DPT, ASTRA and TIGRE](docs/assets/figures/fullscan/fullscan-latency-error.png)

**Forward latency versus image error.** Image relative L2 error is measured against DPT FP64 with 1024 midpoints. DPT and ASTRA use resident inputs; TIGRE includes host transfers. The implementations use different discretisations and timing boundaries.

See [measurement details, raw results and reproduction commands](docs/performance/fullscan.md).

## CT projections

![Native chest CT and computed AP, lateral and oblique projections](docs/assets/figures/release-chest-512/ct_projections.png)

## Install and project a real CT

Use Python 3.12 or later and a CUDA-enabled Torch installation suitable for your GPU. From the source checkout:

```bash
python -m pip install '.[gpu,imaging]'
```

```python
import dpt

ct = dpt.read_volume("chest.nii.gz", units="HU")
mu = dpt.ct_to_attenuation(ct, energy_kev=60.0)
views = dpt.standard_views(ct.grid, ("ap", "lateral"), shape=(512, 512))
images = dpt.project(mu, views, device="cuda", precision="fp32")
transmission = images.neg().exp()
```

`images` is a CUDA tensor with shape `(views, rows, columns)` and dimensionless optical depth. Input values and full physical geometry remain together. NIfTI scaling and DICOM value transforms are applied once.

The installed command saves numerical arrays, PNGs and resolved physical settings:

```bash
dpt project chest.nii.gz --units HU --views ap lateral --energy-kev 60 --output projections
```

Obtain the small real chest and oblique brain samples without adding data to the repository:

```bash
python -m pip install SimpleITK nibabel
python tools/download_samples.py all --data-dir "$HOME/.cache/dpt/samples"
```

## Differentiate a detector region

```python
import torch
from dpt.torch import Projector

field = torch.tensor(mu.values, device="cuda", dtype=torch.float32, requires_grad=True)
projector = Projector(grid=mu.grid, geometry=views, precision="fp32")
prediction = projector(field)
prediction[0, 160:352, 160:352].mean().backward()
volume_sensitivity = field.grad
```

[The differentiation example](examples/differentiate_projection.py) saves sensitivity in the input CT. [Pose fitting](examples/fit_pose.py) accepts measured images and calibration.

## Agentic imaging

Give a coding agent a NIfTI CT or a DICOM series directory. One command prepares two simulated radiographs, fits the CT pose and writes the assessment:

```bash
python examples/agentic_imaging.py run scan.nii.gz --output outputs/ct-run
python examples/agentic_imaging.py run dicom-series/ --output outputs/dicom-run
```

CT values default to HU. A DICOM folder containing several series requires `--series-uid`. The commands return JSON with the case, fit and assessment paths.

For a CT and a reference radiograph, including one from another patient, the [single-image example](docs/applications/agentic-radiograph.md) adjusts the pose to reduce image loss using approximate camera geometry:

```bash
python examples/agentic_radiograph.py inspect scan.nii.gz radiograph.png
python examples/agentic_radiograph.py fit scan.nii.gz radiograph.png \
  --output outputs/radiograph-fit
```

It writes the estimated transform, camera settings, candidate scores, pose trajectory and an image overlay.

### Recorded two-view CT case

This example uses the real **Slicer CTBrain** volume at its native **512 × 507 × 76** resolution, including its oblique orientation. The AP and lateral radiographs are simulated at **256 × 256** pixels. With CTBrain downloaded above:

```bash
mkdir -p outputs
python examples/agentic_imaging.py run \
  "$HOME/.cache/dpt/samples/CTBrain.nii.gz" \
  --source 'Slicer CTBrain / SlicerTestingData' \
  --rights 'Slicer CT-MR Brain donor permission: unrestricted use' \
  --size 256 --samples-per-ray 768 --output outputs/ct-run \
  --max-iterations 120 --max-evaluations 800
```

![AP and lateral projections of the real head CT, initial and fitted residuals, and the recorded convergence curve](docs/assets/figures/agentic-imaging/ct-pose-recovery.png)

The recorded GB10 run reached stationarity in **21 solver evaluations**. RMS displacement across eight geometric reference points fell from **5.208 mm to below 0.001 mm**. Initial and fitted residuals use separate colour scales. The fit uses noise-free simulated images generated with the same projection model.

The commands return JSON for the agent to inspect. A fit that reaches its budget remains unresolved. The [worked guide](docs/applications/agentic-imaging.md) provides the CT source and terms, exact settings, measured results, an agent instruction and the next-view planning interface.

<!-- release:start -->

## Measured performance

NVIDIA RTX PRO 6000 Blackwell Workstation Edition. Complete real chest CT resampled to 256³ with its physical extent preserved. The acquisition uses 360 circular views, 512² detectors and 1024 clipped midpoint samples per ray. Completed resident PyTorch calls, median / p95 milliseconds from three rounds of ten samples. The complete iteration requests both volume and pose gradients.

| Precision | Forward (ms) | Volume VJP (ms) | Pose VJP (ms) | Full iteration (ms) |
|---|---|---|---|---|
| fp32 | 512.60 / 514.35 | 545.79 / 546.97 | 548.91 / 550.70 | 1475.54 / 1478.70 |
| fp64 | 4914.62 / 4920.19 | 4294.51 / 4312.27 | 9481.86 / 9522.05 | 16848.30 / 16869.51 |

See [ASTRA/TIGRE comparisons, errors and Nsight profiles](docs/performance/fullscan.md).

<!-- release:end -->

See [measured boundaries, errors and reproduction commands](docs/performance.md).

## Model and manual

The quick example uses the declared 60 keV water-scaled model `max(1 + HU/1000, 0) * 0.02059 mm^-1`. The water value follows the [NIST mass attenuation table](https://physics.nist.gov/PhysRefData/XrayMassCoef/ComTab/water.html) at density 1 g/cm³. This is a monoenergetic approximation. Already calibrated attenuation can be loaded with `units="mm^-1"`.

1. [Install and make a projection](docs/index.md)
2. [NIfTI, DICOM and physical geometry](docs/geometry.md)
3. [Differentiate in PyTorch](docs/pytorch.md)
4. [Spectrum, detector and transport components](docs/components.md)
5. [Performance and reproduction](docs/performance.md)
6. [API and mathematical reference](docs/reference/index.md)

## Citation

von Csefalvay, C. (2026). *DPT: Differentiable Photon Transport* (version 1.0.0) [Computer software]. [Source repository](https://github.com/hcltech-robotics/dpt).

```bibtex
@software{voncsefalvay2026dpt,
  author = {von Csefalvay, Chris},
  title = {{DPT}: Differentiable Photon Transport},
  year = {2026},
  version = {1.0.0},
  url = {https://github.com/hcltech-robotics/dpt}
}
```

Download [CITATION.cff](CITATION.cff) or [CITATION.bib](CITATION.bib).
