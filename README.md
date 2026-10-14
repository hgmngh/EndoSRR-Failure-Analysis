# EndoSRR mask analysis

Scripts for evaluating specular-reflection masks produced by [EndoSRR](https://github.com/Tobyzai/EndoSRR). The current analysis covers pixel-level segmentation metrics, connected-component sizes, and image differences after inpainting.

This is an exploratory analysis using the two test images included with the available EndoRR dataset, not a benchmark of general model performance.

## Files

| Path | Purpose |
| --- | --- |
| `src/analysis/evaluate_baseline.py` | IoU, Dice, precision, and recall against reference masks |
| `src/analysis/analyze_reflection_scale.py` | Areas of connected components in predicted masks |
| `src/analysis/visualize_reflection_components.py` | Bounding boxes colored by component size |
| `src/analysis/compare_difference.py` | Changed-pixel overlay for restored images |
| `src/baseline/` | Adapted EndoSRR entry point and changes relative to upstream |
| `tests/` | Checks using small synthetic masks |

## Running the analysis

The scripts use Python 3.10 and expect the original EndoSRR directory alongside this repository:

```text
workspace/
  EndoSRR/             # original code, dataset, model outputs
  EndoSRR-Research/    # this repository
```

Install the analysis dependencies and run the scripts from the repository directory:

```bash
python -m pip install -r requirements-analysis.txt
python src/analysis/evaluate_baseline.py
python src/analysis/analyze_reflection_scale.py
python src/analysis/visualize_reflection_components.py
python src/analysis/compare_difference.py
python -m pytest tests/
```

Paths can be overridden with command-line options (`--help` on each script). CSV files and component visualizations are written to `results/`. Difference overlays are written to the original EndoSRR `output/difference/` directory by default.

## Evaluation details

Reference and predicted masks are binarized using pixel values greater than zero. Connected components use 8-connectivity. Component visualization uses 1–10, 11–100, and 101+ pixel groups; these thresholds are exploratory rather than clinically defined. The changed-pixel overlay marks any pixel with a per-channel absolute difference exceeding 30, and does not measure restoration quality.

The reference-mask provenance and correspondence with the predicted masks need further verification. Missing or mismatched inputs now raise an error rather than silently contributing to incomplete results.

## Source and data

The inference script in `src/baseline/` depends on the original EndoSRR project and its pretrained models; it is not a standalone implementation. The MPS adaptation and the Big-LaMa checkpoint loader change are recorded in `src/baseline/patches/`. Only use trusted checkpoint files when deserialization requires `weights_only=False`.

Images, results, and model weights are not included. The adapted upstream code is subject to the original project's GPL-3.0 license; see `LICENSE`.
