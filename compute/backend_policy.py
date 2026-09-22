"""Backend policy, structured fallback warnings and execution protocol."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any
import time


class BackendError(RuntimeError):
    """Base class for infrastructure errors of a compute backend."""


class GPUUnavailableError(BackendError):
    pass


class GPUOutOfMemoryError(BackendError):
    pass


class GPUDeviceLostError(BackendError):
    pass


class GPUCompatibilityError(BackendError):
    pass


GPU_FALLBACK_ERRORS = (
    GPUUnavailableError,
    GPUOutOfMemoryError,
    GPUDeviceLostError,
    GPUCompatibilityError,
)


@dataclass(slots=True)
class BackendWarning:
    code: str
    stage: str
    requested_backend: str
    actual_backend: str
    reason: str
    message: str
    timestamp: float = field(default_factory=time.time)


@dataclass(slots=True)
class StageExecution:
    stage: str
    requested_backend: str
    actual_backend: str
    dtype: str
    fallback: bool = False
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionProtocol:
    requested_backend: str = "auto"
    strict_backend: bool = False
    stages: dict[str, StageExecution] = field(default_factory=dict)
    warnings: list[BackendWarning] = field(default_factory=list)

    def reset(self, *, requested_backend: str | None = None,
              strict_backend: bool | None = None) -> None:
        if requested_backend is not None:
            self.requested_backend = str(requested_backend)
        if strict_backend is not None:
            self.strict_backend = bool(strict_backend)
        self.stages.clear()
        self.warnings.clear()

    def record_stage(self, stage: str, *, actual_backend: str, dtype: str,
                     fallback: bool = False, details: dict[str, Any] | None = None) -> None:
        self.stages[stage] = StageExecution(
            stage=stage,
            requested_backend=self.requested_backend,
            actual_backend=actual_backend,
            dtype=dtype,
            fallback=fallback,
            details=details or {},
        )

    def record_fallback(self, stage: str, *, from_backend: str = "gpu",
                        to_backend: str = "cpu", reason: str,
                        message: str) -> None:
        self.warnings.append(BackendWarning(
            code="BACKEND_FALLBACK",
            stage=stage,
            requested_backend=from_backend,
            actual_backend=to_backend,
            reason=reason,
            message=message,
        ))

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested_backend": self.requested_backend,
            "strict_backend": self.strict_backend,
            "stages": {name: asdict(stage) for name, stage in self.stages.items()},
            "warnings": [asdict(warning) for warning in self.warnings],
        }


def classify_gpu_exception(exc: BaseException) -> BackendError | None:
    """Map known CUDA/CuPy infrastructure failures to stable project errors.

    Unknown exceptions deliberately return None, so algorithm/programming bugs are
    never hidden by an automatic CPU fallback.
    """
    if isinstance(exc, GPU_FALLBACK_ERRORS):
        return exc

    if isinstance(exc, (ModuleNotFoundError, ImportError)) and any(
        name in str(exc).lower() for name in ("cupy", "cuda", "cupyx")
    ):
        return GPUUnavailableError(str(exc))

    cls_name = exc.__class__.__name__.lower()
    module = exc.__class__.__module__.lower()
    text = str(exc).lower()
    signature = f"{module}.{cls_name} {text}"

    if "outofmemory" in cls_name or "out of memory" in text or "memory allocation" in text:
        return GPUOutOfMemoryError(str(exc))

    compatibility_markers = (
        "cudaerrorinsufficientdriver", "insufficient driver", "driver version is insufficient",
        "invalid device function", "no kernel image", "unsupported ptx", "nvrtc",
        "cudart", "cuda driver", "runtime version",
    )
    if any(marker in signature for marker in compatibility_markers):
        return GPUCompatibilityError(str(exc))

    device_lost_markers = (
        "devicelost", "device lost", "device-side assert", "launch failure",
        "context is destroyed", "context destroyed",
    )
    if any(marker in signature for marker in device_lost_markers):
        return GPUDeviceLostError(str(exc))

    unavailable_markers = (
        "cudadeviceerror", "nodevice", "no cuda-capable device", "cuda unavailable",
        "gpu processor not available", "gpu not available", "cuda initialization",
        "initialization error",
    )
    if any(marker in signature for marker in unavailable_markers):
        return GPUUnavailableError(str(exc))

    # CuPy/CUDA runtime errors not covered above are infrastructure errors, but
    # ValueError/IndexError/etc. from our own algorithm are intentionally excluded.
    if ("cupy.cuda" in module or "cuda.runtime" in module) and not isinstance(
        exc, (ValueError, TypeError, IndexError, KeyError, AssertionError)
    ):
        return GPUUnavailableError(str(exc))

    return None


def fallback_or_raise(protocol: ExecutionProtocol | None, stage: str,
                      exc: BaseException) -> BackendError:
    mapped = classify_gpu_exception(exc)
    if mapped is None:
        raise exc
    if protocol is not None and protocol.strict_backend:
        raise mapped from exc
    if protocol is not None:
        protocol.record_fallback(
            stage,
            reason=mapped.__class__.__name__,
            message=str(mapped),
        )
    return mapped
