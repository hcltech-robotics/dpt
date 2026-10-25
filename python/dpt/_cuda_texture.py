"""Compile and launch the optional CUDA midpoint kernels."""

from __future__ import annotations

import ctypes as ct
import hashlib
import os
import shutil
import subprocess
import tempfile
import threading
import weakref
from functools import cache
from pathlib import Path
from typing import Any

from dpt.contracts import ContractError


class _Configuration(ct.Structure):
    _fields_ = [
        *((name, ct.c_int) for name in ("nx", "ny", "nz", "width", "pixels", "samples")),
        ("object_to_grid", ct.c_double * 9),
        *(
            (name, ct.c_double * 3)
            for name in ("grid_origin", "source", "detector_origin", "column_step", "row_step")
        ),
        ("ray_bundle", ct.c_int),
    ]


@cache
def _library(architecture: int) -> Any:
    compiler = os.environ.get("DPT_NVCC") or shutil.which("nvcc")
    if compiler is None:
        raise ContractError("cuda_texture requires nvcc; set DPT_NVCC to the CUDA compiler")
    source = Path(__file__).with_name("native")
    files = [source / name for name in ("common.cuh", "texture.cu", "footprint.cu")]
    version = subprocess.run([compiler, "--version"], check=True, capture_output=True).stdout
    digest = hashlib.sha256(version + str(architecture).encode())
    for path in files:
        digest.update(path.read_bytes())
    directory = Path(os.environ.get("DPT_CUDA_CACHE", Path.home() / ".cache/dpt/cuda"))
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"midpoint-{digest.hexdigest()[:24]}.so"
    if not target.exists():
        # Each compiler writes a private file; concurrent builds publish atomically.
        with tempfile.TemporaryDirectory(prefix="build-", dir=directory) as temporary:
            built = Path(temporary) / "midpoint.so"
            subprocess.run(
                [
                    compiler,
                    "--shared",
                    "-Xcompiler",
                    "-fPIC",
                    "-O3",
                    "-lineinfo",
                    "--std=c++17",
                    "--ftz=false",
                    "--prec-div=true",
                    "--prec-sqrt=true",
                    f"-arch=sm_{architecture}",
                    str(files[1]),
                    str(files[2]),
                    "-o",
                    str(built),
                ],
                check=True,
            )
            os.replace(built, target)
    library = ct.CDLL(str(target))
    pointer, integer = ct.c_void_p, ct.c_int
    config = ct.POINTER(_Configuration)
    library.dpt_cuda_error.argtypes = [integer]
    library.dpt_cuda_error.restype = ct.c_char_p
    library.dpt_texture_create.argtypes = [pointer, ct.c_size_t, integer, ct.POINTER(ct.c_uint64)]
    library.dpt_texture_destroy.argtypes = [ct.c_uint64, integer, pointer]
    library.dpt_atlas_create.argtypes = [config, integer, ct.POINTER(pointer)]
    library.dpt_atlas_destroy.argtypes = [pointer, integer, pointer]
    library.dpt_atlas_bytes.argtypes = [pointer]
    library.dpt_atlas_bytes.restype = ct.c_size_t
    library.dpt_forward.argtypes = [
        config,
        pointer,
        ct.c_uint64,
        pointer,
        pointer,
        pointer,
        pointer,
        pointer,
        integer,
        pointer,
        integer,
        pointer,
    ]
    library.dpt_scratch_bytes.argtypes = [config]
    library.dpt_scratch_bytes.restype = ct.c_size_t
    library.dpt_forward_many.argtypes = [
        config,
        pointer,
        integer,
        pointer,
        ct.c_uint64,
        pointer,
        pointer,
        pointer,
        pointer,
        integer,
        pointer,
    ]
    library.dpt_adjoint.argtypes = [
        config,
        pointer,
        pointer,
        pointer,
        pointer,
        integer,
        pointer,
        integer,
        integer,
        pointer,
        pointer,
        integer,
        pointer,
    ]
    return library


def _release_texture(
    library: Any, handle: int, device: int, stream: int, owners: tuple[Any, Any]
) -> int:
    # Keep the allocation and owning stream alive through the completion fence.
    return library.dpt_texture_destroy(handle, device, stream)


class _Texture:
    def __init__(self, owner: NativeProjection, field: Any):
        self.field = field
        self.handle = ct.c_uint64()
        owner.check(
            owner.library.dpt_texture_create(
                field.ptr, field.size, owner.device, ct.byref(self.handle)
            )
        )
        self._release = weakref.finalize(
            self,
            _release_texture,
            owner.library,
            self.handle.value,
            owner.device,
            owner.stream,
            (owner.context, field),
        )
        self._release.atexit = False

    def close(self, owner: NativeProjection) -> None:
        result = self._release()
        if result is not None:
            owner.check(result)


def _release_atlas(library: Any, handle: Any, device: int, stream: int, context: Any) -> int:
    return library.dpt_atlas_destroy(handle, device, stream)


class _Atlas:
    def __init__(self, owner: NativeProjection):
        self.handle = ct.c_void_p()
        owner.check(
            owner.library.dpt_atlas_create(
                ct.byref(owner.configuration), owner.device, ct.byref(self.handle)
            )
        )
        self.nbytes = owner.library.dpt_atlas_bytes(self.handle)
        self.lock = threading.Lock()
        self._release = weakref.finalize(
            self,
            _release_atlas,
            owner.library,
            self.handle,
            owner.device,
            owner.stream,
            owner.context,
        )
        self._release.atexit = False


_atlases: weakref.WeakValueDictionary[tuple[Any, ...], _Atlas] = weakref.WeakValueDictionary()
_atlas_lock = threading.Lock()


def _shared_atlas(owner: NativeProjection) -> _Atlas:
    config = owner.configuration
    key = (owner.library._name, owner.device, owner.stream, config.nx, config.ny, config.nz)
    with _atlas_lock:
        atlas = _atlases.get(key)
        if atlas is None:
            atlas = _Atlas(owner)
            _atlases[key] = atlas
        return atlas


class NativeProjection:
    """Own texture lifetime and adjoint scratch on one workspace stream."""

    def __init__(self, workspace: Any):
        context = workspace.context
        self.context = context
        self.library = _library(context.device.arch)
        self.device = context.device.ordinal
        self.stream = context.stream.cuda_stream
        self.configuration = _Configuration()
        config = workspace._configuration
        self.configuration.nx, self.configuration.ny, self.configuration.nz = config.shape
        for name in ("width", "pixels", "samples"):
            setattr(self.configuration, name, getattr(config, name))
        self.configuration.object_to_grid[:] = [
            config.object_to_grid[row, column] for row in range(3) for column in range(3)
        ]
        for name in ("grid_origin", "source", "detector_origin", "column_step", "row_step"):
            getattr(self.configuration, name)[:] = list(getattr(config, name))
        self.configuration.ray_bundle = config.is_ray_bundle
        self.sources = config.ray_sources
        self.endpoints = config.ray_endpoints
        size = (
            self.library.dpt_scratch_bytes(ct.byref(self.configuration))
            if workspace.spec.active_volume
            else 0
        )
        with context.scope():
            self.scratch = context.wp.empty(size, dtype=context.wp.uint8, device=context.device)
        self.texture: _Texture | None = None
        self.atlas: _Atlas | None = None

    @property
    def scratch_bytes(self) -> int:
        # Workspaces on the same stream and grid share the atlas allocation.
        return int(self.scratch.size) + (self.atlas.nbytes if self.atlas is not None else 0)

    def check(self, code: int) -> None:
        if code:
            detail = self.library.dpt_cuda_error(code).decode()
            raise RuntimeError(f"CUDA midpoint kernel: {detail}")

    def resources(self, field: Any) -> tuple[_Texture, _Atlas]:
        if self.texture is None or self.texture.field.ptr != field.ptr:
            previous, self.texture = self.texture, None
            if previous is not None:
                previous.close(self)
            self.texture = _Texture(self, field)
        if self.atlas is None:
            self.atlas = _shared_atlas(self)
        return self.texture, self.atlas

    def forward(self, field: Any, pose: Any, output: Any, double_output: bool, status: Any) -> None:
        texture, atlas = self.resources(field)
        # Keep each pack/launch pair together when host threads share the stream.
        # Every call repacks the field, including mutations to an existing pointer.
        with atlas.lock:
            self.check(
                self.library.dpt_forward(
                    ct.byref(self.configuration),
                    field.ptr,
                    texture.handle.value,
                    atlas.handle,
                    pose.ptr,
                    self.sources.ptr,
                    self.endpoints.ptr,
                    output.ptr,
                    int(double_output),
                    status.ptr,
                    self.device,
                    self.stream,
                )
            )

    def adjoint(
        self,
        pose: Any,
        seed: Any,
        output: Any,
        double_seed: bool,
        double_output: bool,
        accumulate: bool,
        status: Any,
    ) -> None:
        self.check(
            self.library.dpt_adjoint(
                ct.byref(self.configuration),
                pose.ptr,
                self.sources.ptr,
                self.endpoints.ptr,
                seed.ptr,
                int(double_seed),
                output.ptr,
                int(double_output),
                int(accumulate),
                status.ptr,
                self.scratch.ptr,
                self.device,
                self.stream,
            )
        )


class NativeBatch:
    """Refresh one texture and project a fixed set of cone views."""

    def __init__(self, owners: tuple[NativeProjection, ...]):
        if not 1 <= len(owners) <= 65535:
            raise ContractError("CUDA projection batches require 1 to 65535 views")
        self.owners = owners
        self.first = owners[0]
        self.count = len(owners)
        reference = self.first.configuration
        shape = (reference.nx, reference.ny, reference.nz, reference.pixels, reference.samples)
        if any(
            owner.device != self.first.device
            or owner.stream != self.first.stream
            or owner.configuration.ray_bundle
            or tuple(
                getattr(owner.configuration, name)
                for name in ("nx", "ny", "nz", "pixels", "samples")
            )
            != shape
            for owner in owners
        ):
            raise ContractError("CUDA projection batches require matching grids and one stream")
        packed = (_Configuration * self.count)(*(owner.configuration for owner in owners))
        context = self.first.context
        with context.scope():
            self.configurations = context.wp.array(
                list(bytes(packed)), dtype=context.wp.uint8, device=context.device
            )

    def forward(self, field: Any, poses: Any, output: Any, statuses: Any) -> None:
        owner = self.first
        texture, atlas = owner.resources(field)
        with atlas.lock:
            owner.check(
                owner.library.dpt_forward_many(
                    ct.byref(owner.configuration),
                    self.configurations.ptr,
                    self.count,
                    field.ptr,
                    texture.handle.value,
                    atlas.handle,
                    poses.ptr,
                    output.ptr,
                    statuses.ptr,
                    owner.device,
                    owner.stream,
                )
            )
