import json
import os
import subprocess
import sys
import time
from pathlib import Path


LOGS = Path("/logs/verifier")
TESTS = Path("/tests")

WEIGHTS = {
    "1": 25,
    "2": 20,
    "3": 15,
    "4": 25,
    "5": 15,
}

TEST_FILES = {
    "1": "test_visible_validation.py",
    "2": "test_calibration_robustness.py",
    "3": "test_numerical_robustness.py",
    "4": "test_hidden_generalization.py",
    "5": "test_scientific_report.py",
}


def run_criterion(number, filename):
    start = time.monotonic()

    environment = os.environ.copy()
    environment["TASK_WORKSPACE"] = "/app/submission"
    environment["HIDDEN_DATA_PATH"] = (
        "/tests/fixtures/reference_hidden_validation.csv"
    )

    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            str(TESTS / filename),
        ],
        cwd="/app",
        env=environment,
        capture_output=True,
        text=True,
        timeout=300,
    )

    elapsed = time.monotonic() - start

    output = (process.stdout + process.stderr).strip()

    return {
        "passed": process.returncode == 0,
        "output": output[-12000:],
        "seconds": round(elapsed, 3),
    }


def main():
    LOGS.mkdir(parents=True, exist_ok=True)

    outcomes = {}
    diagnostics = {}

    for number, filename in TEST_FILES.items():
        try:
            result = run_criterion(number, filename)
            outcomes[number] = 1 if result["passed"] else 0
            diagnostics[number] = result
        except subprocess.TimeoutExpired:
            outcomes[number] = 0
            diagnostics[number] = {
                "passed": False,
                "output": "Criterion exceeded the 300-second verifier limit.",
                "seconds": 300,
            }
        except Exception as error:
            outcomes[number] = 0
            diagnostics[number] = {
                "passed": False,
                "output": f"{type(error).__name__}: {error}",
            }

    weighted_score = sum(
        WEIGHTS[number] * outcomes[number]
        for number in WEIGHTS
    ) / 100

    reward = int(all(outcomes.values()))

    report = {
        "task_id": "thermal-interface-identification",
        "reward": reward,
        "weighted_score": weighted_score,
        "per_criterion": outcomes,
        "full_pass": bool(reward),
        "weights": WEIGHTS,
        "diagnostics": diagnostics,
    }

    (LOGS / "criteria.json").write_text(
        json.dumps(report, indent=2, allow_nan=False)
    )

    (LOGS / "reward.txt").write_text(str(reward))

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
