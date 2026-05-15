# Changelog Entry

## Added
- New CLI entrypoint `prediction-bot`
- Module entrypoint support via `python3 -m prediction_bot`
- CLI commands for training, prediction, and evaluation
- Optional JSON output file support for CLI workflows
- Batch prediction support via `Predictor.predict_many()`
- Root project `README.md`
- Release checklist document

## Changed
- Renamed package from `src` to `prediction_bot`
- Updated project packaging and coverage configuration
- Improved temporal splitting to use date-boundary partitioning
- Hardened Pydantic schemas with tighter validation constraints
- Expanded evaluation output with calibration information
- Optimized training label generation
- Optimized internal data pipeline conversion paths
- Updated documentation to reflect new package layout and CLI usage

## Fixed
- Reduced ambiguity and fragility in package import structure
- Improved temporal integrity guarantees for train/val/test partitioning
- Closed documentation/spec drift around evaluation capabilities
- Added smoke coverage for the module entrypoint

## Validation
- Test suite passing: **72 passed**
- Coverage: **97%**
