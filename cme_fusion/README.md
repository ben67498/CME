# CME Fusion Smoke Pipeline

Tiny end-to-end PyTorch pipeline for LASCO + SDO fusion binary CME classification.

## Quickstart (Colab)
1. Open `colab_smoke.ipynb` in Colab.
2. Run cells in order:
   - install dependencies
   - run tests
   - run smoke test

Or locally:
```bash
pip install -r requirements.txt
pytest -q
python -m src.smoke_test --run_name smoke
```

## What smoke test does
- downloads 3 SDO infer_data netCDF files from HF dataset `nasa-ibm-ai4science/core-sdo`
- downloads LASCO C2 sample for 2011-01-20 (24 files)
- downloads CDAW `univ_all.txt` (2 URL fallback)
- aligns by time, creates labels, trains 2 epochs
- writes `runs/smoke/metrics.json`, `runs/smoke/best.pt`, and debug image `runs/smoke/debug_grid.png`

## Scaling to full core-sdo
For larger runs, list available files by date range from S3 mirror and select subsets:
```bash
aws s3 ls s3://nasa-ibm-ai4science/core-sdo/ --no-sign-request
```
Then feed many `.nc` paths into your own training script. Recommended: downsample/crop (e.g., 256/512) and optionally use temporal difference images to reduce static background effects.

## Notes
- Deterministic seeds are set.
- LASCO download has two backend code paths with fallback.
- CDAW parse is resilient to mixed formatting and optional numeric columns.
