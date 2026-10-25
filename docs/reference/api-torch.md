# Torch API

Use [`dpt.torch.Projector`](#prepared-projection-module) for a prepared multiview calculation. The [real-CT tutorial](../pytorch.md) covers that path. The [Torch guide](../guides/torch.md) covers lower-level projection, detector operators and transport gradients. Expected-transport interfaces live in `dpt.interop.torch_transport`.

## Projection and rigid pose

`compose_pose` maps six local-right SE(3) coordinates to the twelve-value FP64 pose expected by projection. The attenuation field is FP32. The output precision and active derivatives follow the prepared `ProjectionSpec`.


::: dpt.torch.geometry.compose_pose
    options:
      show_source: true

::: dpt.interop.torch.project_optical_depth

## Material and spectral derivatives

`project_spectral_signal(fields, pose, coefficients, weights, response, ...)` returns a Torch-owned detector image. Its material and spectral workspaces share device, stream, material count and retained precision. `SpectralSpec.output_domain` chooses the linear mean or its natural logarithm.

| Input | Contiguous layout | Storage and units | Prepared activity |
|---|---|---|---|
| Fields | `(M, nz, ny, nx)` or `(M, voxels)` | FP32, dimensionless | `ProjectionSpec.active_volume` |
| Pose | `(12,)` | FP64, rotation then translation in mm | `ProjectionSpec.active_pose` |
| Coefficients | `(M, K)` | FP32, mm⁻¹ | `SpectralSpec.active_coefficients` |
| Weights | Shared `(K,)` or pixel-specific `(K, P)` | FP32, integrated photon populations | `SpectralSpec.active_weights` |
| Response | Shared `(K,)` or pixel-specific `(K, P)` | FP32, declared response per photon | `SpectralSpec.active_response` |

Coefficient and factor arrays also accept contiguous flat views with the same element order. Shared layouts follow `shared_weights` and `shared_response`. A tensor with `requires_grad=True` requires its activity flag. Field or pose derivatives additionally require `active_paths=True` and a prepared material path-cotangent buffer. The [worked spectral example](../guides/torch.md#differentiate-a-spectral-image) prepares this complete chain.

Gradients preserve each input's shape and dtype. Pose backward uses twelve native storage partials, so a pose created by `compose_pose` receives the full chart chain rule. Each node saves all five input tensors and recomputes its paths before reverse evaluation. Distinct outstanding forwards can reuse native scratch sequentially. Keep each node's saved tensors unchanged through its backward.

Call `project_spectral_signal` with the prepared material and spectral workspaces. Torch modules can register the supplied calibration tensors as buffers or parameters according to the application.


::: dpt.interop.torch.project_spectral_signal
    options:
      show_source: true

## Expected transport and score gradients

`estimate_expected_transport(log_voxel_density, log_material_density, source_amplitude, *, workspace, estimator)` returns an `ExpectedTransportEstimate`. Its `signal` is a Torch-owned FP64 image with the detector's `(rows, columns)` shape. The other fields retain the resolved batch/source identities, parameter specification and native transport specification. `expected_transport_signal` takes the same arguments and returns the image alone.

Prepare the fixed native model with `prepare_expected_transport(native_workspace, *, source, parameters=None, slots_per_history=64, max_snapshot_bytes=256 * 1024 * 1024)`. The source is a fixed `ParallelBeam` or `PointConeSource`. Preparation copies material IDs and coefficient tables. Each forward supplies the density multipliers and amplitude for its own model snapshot.

| Input | Contiguous FP64 shape | Meaning |
|---|---|---|
| `log_voxel_density` | Grid shape or `(voxels,)`, x fastest | Log of the dimensionless voxel multiplier |
| `log_material_density` | `(materials,)` | Log of the dimensionless material multiplier |
| `source_amplitude` | Scalar or `(1,)` | Direct nonnegative measurement multiplier |

The physical voxel and material multipliers are `exp(log_density)`. Both must remain finite and positive, including fixed entries. The direct source-amplitude derivative is defined at zero. The voxel tensor supplies the complete voxel multiplier field. Preparation uses the native workspace's fixed geometry, material IDs and coefficients.

`TransportAutogradSpec` names these three coordinate systems explicitly. `active_voxels=None` and `active_materials=None` select all entries, a tuple selects those IDs, and `()` fixes the whole chart. Unselected entries receive zero gradients. Set `active_source_amplitude=False` to fix the amplitude. A tensor with `requires_grad=True` requires a selected chart. Gradients retain each tensor's shape and FP64 dtype.

`ExpectedTransportEstimator(forward_batch, derivative_batch, cotangent_origin="forward-batch", external_cotangent_batch=None)` declares the histories used by the forward and score calculations:

| Cotangent origin | Meaning | Independence checked |
|---|---|---|
| `"forward-batch"` | The upstream cotangent is computed from the forward image | Derivative batch from forward batch |
| `"deterministic"` | The cotangent is fixed independently of the simulated histories | Derivative batch from forward batch |
| `"external-batch"` | The cotangent uses the named `external_cotangent_batch` | Derivative batch from both named batches |

The adapter attaches the library source-sampler tokens and checks transport and source-draw addresses. Give independent draws disjoint addresses. Changing a namespace label leaves their identity intact.

For forward batch `A`, derivative batch `B` and upstream cotangent `c`, the operation is

```text
forward:  mu_hat_A = complete original-history signal mean
backward: J_hat_B.T @ c
```

`J_hat_B` estimates the Jacobian of the expected signal with the native likelihood/direct-weight score. Forward divides by `A.count`, and backward divides by `B.count`, including misses in both denominators. With a composed loss, `c(A) = grad L(mu_hat_A)`. Independence removes the signal/score covariance. An affine cotangent, including a squared-error residual, gives the corresponding independent-product gradient. A nonlinear cotangent retains the sampling bias of evaluating `grad L` at a finite-sample image. Choose the loss estimator with that distinction in mind. The [worked example](../guides/torch.md#differentiate-an-expected-transport-signal) uses squared error.

Call `estimate_expected_transport` with the current tensors and estimator for each iteration.

Each outstanding node owns its density/amplitude snapshot, source states and native derivative scratch. Keep all original inputs unchanged through backward, including fixed inputs. The memory budget covers the complete planned node, including all selected material/source groups and sparse voxel-score storage. `slots_per_history` bounds the distinct active voxel entries retained for a history. Exhausted history or sparse-storage budgets raise `IncompleteHistoryError`. Discard that estimate and increase the appropriate budget before evaluating again. Calls check native status on the owning stream. The adapter supports first-order score derivatives with `backward()` or `torch.autograd.grad` settings. Use `torch.no_grad()` for a signal-only calculation.


::: dpt.interop.torch_transport.TransportAutogradSpec

::: dpt.interop.torch_transport.ExpectedTransportEstimator

::: dpt.interop.torch_transport.prepare_expected_transport

::: dpt.interop.torch_transport.estimate_expected_transport

::: dpt.interop.torch_transport.expected_transport_signal

## Prepared projection module

`Projector(grid=..., geometry=..., precision="fp32", samples_per_ray=1024)` fixes the acquisition and prepares native device workspaces lazily on first use. `module(field, pose=None)` returns `(views, rows, columns)` optical depth. The field is a contiguous FP32 tensor with the grid's `(nz, ny, nx)` shape, containing finite nonnegative attenuation in mm⁻¹. Output is FP32 for `precision="fp32"` and FP64 for `precision="fp64"`. The input field and its gradient remain FP32 in both cases. `pose` holds FP64 right SE(3) increments `(6,)` or `(views, 6)`. `packed_pose=` explicitly supplies `(12,)` or `(views, 12)` transforms instead. Both pose forms describe object-to-world motion. `pose` acts about the constructor's `anchor` (identity by default), while `packed_pose` overrides it. Supply only one form. A tensor anchor must be on the field device, using `module.to(device)` when needed. See [the real-volume Torch example](../pytorch.md).

All views share the detector shape and resolved midpoint count. `step_mm` is an alternative to `samples_per_ray`: it resolves the count once from the volume-support diagonal, and each clipped ray uses that count. The module supports fixed cone, parallel and curved detector views. Its functional dispatcher operator takes serialisable geometry values and returns a fresh tensor. A separately registered VJP computes first-order gradients. FakeTensor metadata implementations let `torch.compile(..., fullgraph=True)` trace the surrounding tensor graph. Use `torch.compile(module, backend="inductor", fullgraph=True)` for Inductor. The default backend uses Warp projection kernels. Torch `>=2.14,<3` is the supported dependency.

Field, pose and cotangent validation is enqueued on the native execution stream. A single aggregate status read completes each successful forward or backward call. Failed calls may read individual statuses again to deliver view-specific diagnostics. Forward and backward each have their own completion boundary. Mutable inputs are scanned on every invocation. Final accumulated volume gradients are checked for overflow. Delivered errors clear the private status state for the next call. Outputs from a failed call must be discarded.

`out=` declares mutation of the output alone and is restricted to inference. It must be contiguous, have the projected shape/dtype/device, not overlap an input and have `requires_grad=False`. When an input requires gradients, use `torch.no_grad()` or omit `out=`. The method returns the supplied buffer. Functional outputs and gradients never alias inputs or private scratch. Saved tensors retain Torch version checks. Checked operators are marked `cudagraph_unsafe` and do not support CUDA graph capture or higher-order differentiation. Native workspace caching is bounded and distinguishes geometry, device, current calling stream and derivative activity. Native launches retain tensor storage through completion. Separate outstanding autograd nodes keep their own saved inputs and outputs. Follow Torch's producer/consumer stream dependencies when crossing streams. See [full-scan measurements](../performance/fullscan.md) for the checked timing boundaries.

::: dpt.torch.modules.Projector

Set `backend="cuda_texture"` to select the CUDA C++ midpoint forward and volume
adjoint with `precision="fp32"`. The backend retains the FP32 sample positions
and interpolation weights. Pose derivatives, FP64 projection and CPU execution
use Warp. The default backend is `"warp"`.

Multiview cone projections run as one CUDA launch and refresh the volume texture
once per call. Scalar calls refresh it independently. Workspaces with the same
grid and stream share the texture allocation.

The first CUDA preparation compiles for the selected GPU with `nvcc`. Set
`DPT_NVCC` to a compiler path if it is absent from `PATH`. Compiled libraries are
cached in `~/.cache/dpt/cuda`; `DPT_CUDA_CACHE` selects another cache directory.

## Transmission, detector and objective

These functions return Torch-owned values and apply the native first-order VJPs in backward. Workspace specifications select active optical depth, beam and shared calibration inputs. Objective observations, weights and masks remain fixed.


::: dpt.torch.operators.transmit

::: dpt.torch.operators.affine_calibrate

::: dpt.torch.operators.blur

::: dpt.torch.operators.objective

## Tensor and stream lifetimes

Keep each forward node's inputs unchanged until its reverse calculation completes. Outputs own their storage. Projection and spectral nodes can reuse workspace scratch sequentially. Expected-transport nodes retain their own model snapshots and derivative scratch. The adapters use CUDA events to order the calling and owning streams and `record_stream` to retain borrowed tensor storage through consumption. Follow Torch's producer-stream rules before passing an input created on another stream.

Read [execution and ownership](../concepts/execution.md) for numerical completion and [the Torch guide](../guides/torch.md#retain-tensors-through-backward) for first-order autograd use.
