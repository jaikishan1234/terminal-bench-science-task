#!/bin/sh
set -eu

WORKSPACE="${TASK_WORKSPACE:-/workspace}"

export TASK_WORKSPACE="$WORKSPACE"

pytest tests/test_visible_validation.py -q
pytest tests/test_calibration_robustness.py -q
pytest tests/test_numerical_robustness.py -q
pytest tests/test_hidden_generalization.py -q
pytest tests/test_scientific_report.py -q
