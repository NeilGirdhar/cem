"""Typst compilation and PDF viewer launching."""

import os
import shutil
import subprocess  # ruff:ignore[suspicious-subprocess-import]
from pathlib import Path

_TYPST_SOURCE = Path("results/render.typ")


def compile_pdf(json_path: Path, pdf_path: Path) -> None:
    """Compile ``json_path`` into ``pdf_path`` using the thesis typst template."""
    typst = shutil.which("typst")
    if typst is None:
        msg = "Could not find 'typst' on PATH."
        raise SystemExit(msg)
    subprocess.run(  # ruff:ignore[subprocess-without-shell-equals-true]
        [
            typst,
            "compile",
            "--input",
            f"source={json_path.name}",
            str(_TYPST_SOURCE),
            str(pdf_path),
        ],
        check=True,
    )


def open_pdf(pdf_path: Path, viewer: str | None = None) -> None:
    """Open ``pdf_path`` in ``viewer`` (defaults to the platform's handler)."""
    command = viewer or _default_viewer()
    tool = shutil.which(command)
    if tool is None:
        msg = f"Could not find viewer '{command}' on PATH."
        raise SystemExit(msg)
    subprocess.Popen(  # ruff:ignore[subprocess-without-shell-equals-true]
        [tool, str(pdf_path)],
        start_new_session=True,
    )


def _default_viewer() -> str:
    if os.name == "nt":
        return "start"
    if os.uname().sysname == "Darwin":
        return "open"
    return "xdg-open"
