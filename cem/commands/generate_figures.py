import json
import shutil
import subprocess  # noqa: S404
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from cem.structure import Demo
from cem.structure.solution import InferenceResults, TrainingResults

_TYPST_DIR = Path("typst")
_TYPST_SOURCE = _TYPST_DIR / "render.typ"


def generate_figures(
    demo: Demo,
    labeled_results: Sequence[tuple[str, tuple[TrainingResults, InferenceResults]]],
    *,
    display: bool,
) -> None:
    result: dict[str, dict[str, Any]] = {}
    for plotter in demo.plotters():
        plot_key = plotter.name
        plot_data: dict[str, Any] = {}
        line_plots: dict[str, str] = {}
        for label, results in labeled_results:
            variant_data = plotter.plotted_series(results[0], results[1], label)
            line_plot_titles = plotter.line_plot_titles(label)
            for key, values in variant_data.items():
                output_key = key if key == "iteration" or not label else f"{label}.{key}"
                plot_data[output_key] = values
                if key in line_plot_titles:
                    line_plots[output_key] = line_plot_titles[key]
        plot_data["line plots"] = line_plots
        result[plot_key] = plot_data

    json_path = _TYPST_DIR / f"{demo.name}.json"
    pdf_path = _TYPST_DIR / f"{demo.name}.pdf"
    json_path.parent.mkdir(exist_ok=True)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")

    typst = shutil.which("typst")
    if typst is None:
        msg = "Could not find 'typst' on PATH."
        raise SystemExit(msg)
    subprocess.run(  # noqa: S603
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

    if display:
        zathura = shutil.which("zathura")
        if zathura is None:
            msg = "Could not find 'zathura' on PATH."
            raise SystemExit(msg)
        subprocess.Popen(  # noqa: S603
            [zathura, str(pdf_path)],
            start_new_session=True,
        )
