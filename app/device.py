"""Where the models run.

Ingest is the only expensive part of this project and it all happens on
whoever's machine builds the corpus -- transcription, then a CTC alignment pass
over every word clip. Serving, by contrast, is ffmpeg cutting 480x270 clips and
needs no accelerator at all. So the container installs torch from the CPU wheel
index on purpose (a CUDA build drags in gigabytes of driver payload a headless
box never uses), and this module exists so that choice does not also force the
ingest machine onto its CPU.

"auto" is the default everywhere: CUDA when a usable one is present, otherwise
CPU. That resolves to CPU inside the container with no configuration, and to
the GPU on a workstation that has one, which is exactly the split we want.
MRS_DEVICE overrides it for the awkward cases -- a GPU busy with something
else, or a driver mismatch that makes CUDA present but not usable.
"""

from __future__ import annotations

import logging
import os

log = logging.getLogger(__name__)

CHOICES = ("auto", "cuda", "cpu")

_resolved: str | None = None


def _cuda_ok() -> bool:
    """True only if CUDA is importable, visible, and actually runs.

    is_available() alone is not enough. A CPU-only wheel returns False, which is
    fine, but the nastier case is a CUDA wheel against a driver it cannot talk
    to: is_available() can still say True and every later call dies mid-ingest,
    hours in. A one-off tensor round-trip settles it in milliseconds.
    """
    try:
        import torch
    except Exception:
        return False
    if not torch.cuda.is_available():
        return False
    try:
        torch.zeros(1).cuda().cpu()
        return True
    except Exception as exc:
        log.warning("CUDA present but unusable, falling back to CPU: %s", exc)
        return False


def resolve(requested: str | None = None) -> str:
    """Turn auto/cuda/cpu (or MRS_DEVICE, or nothing) into a real device."""
    want = (requested or os.environ.get("MRS_DEVICE") or "auto").lower()
    if want not in CHOICES:
        raise ValueError(f"device must be one of {', '.join(CHOICES)}, got {want!r}")

    if want == "cpu":
        return "cpu"
    if want == "cuda":
        # Asked for explicitly, so a fallback would be a silent downgrade that
        # turns a two-hour ingest into a two-day one without saying so.
        if not _cuda_ok():
            raise RuntimeError(
                "--device cuda was asked for but no usable CUDA device is present.\n"
                "  torch may be the CPU wheel: check torch.version.cuda is not None.\n"
                "  Use --device auto to fall back to the CPU instead of failing."
            )
        return "cuda"
    return "cuda" if _cuda_ok() else "cpu"


def get(requested: str | None = None) -> str:
    """resolve(), remembered, so a long run reports the device once."""
    global _resolved
    if _resolved is None:
        _resolved = resolve(requested)
        log.info("compute device: %s", describe(_resolved))
    return _resolved


def describe(device: str | None = None) -> str:
    """A human line naming the device, for the top of an ingest log."""
    device = device or resolve()
    if device != "cuda":
        return "cpu"
    try:
        import torch
        name = torch.cuda.get_device_name(0)
        gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        return f"cuda ({name}, {gb:.1f} GB)"
    except Exception:
        return "cuda"


def add_argument(parser) -> None:
    """The --device flag, worded the same way everywhere it appears."""
    parser.add_argument(
        "--device",
        default=None,
        choices=list(CHOICES),
        help="Where to run the models: auto (default -- CUDA if usable, else "
             "CPU), cuda, or cpu. Overrides $MRS_DEVICE.",
    )
