# Thermal Interface Identification

You are given a Python project for investigating transient heat transfer through a wall made from two different materials.

The project contains three experimental datasets:

- `data/heating.csv`
- `data/cooling.csv`
- `data/moderate_heating.csv`

The existing implementation in `src/` models the wall and calibrates the thermal conductivities of the two materials. The current model does not reproduce the experimental measurements accurately.

## Objective

Investigate the model, experimental data, and calibration procedure, then improve the implementation so that it provides a physically meaningful and numerically stable explanation of the observed temperature behavior.

Do not hardcode expected answers or dataset-specific outputs.

## Required work

1. Inspect the existing source code and all three datasets.
2. Run the existing implementation and quantify where and how its predictions disagree with the measurements.
3. Determine an appropriate physical and numerical improvement to the model based on the evidence in the data.
4. Extend the parameter estimation procedure so that all required physical parameters can be calibrated from the provided experiments.
5. Ensure calibrated physical parameters are finite, positive, and reproducible.
6. Validate the improved model against all three visible experiments using quantitative error metrics.
7. Check that the numerical solver remains stable and produces finite, physically sensible temperatures under the provided experiments.
8. Demonstrate that the calibrated model can make predictions for an operating condition not present in the visible datasets.
9. Create a scientific report in `report.md` describing the investigation, physical interpretation, calibration procedure, numerical method, validation results, units, and reproducibility considerations.

## Constraints

- Use only the data and software available in the task environment.
- Do not use network services or external datasets.
- Do not embed the expected hidden results in the implementation.
- Preserve the existing dataset format unless there is a strong technical reason to extend it.
- The final implementation must remain usable through the existing Python APIs where practical.
- The solution should represent a genuine physical model rather than merely fitting or transforming the observed values.

## Completion criteria

The task is complete when the improved implementation, calibration procedure, validation evidence, and scientific report are present in the workspace and the resulting model demonstrates substantially improved agreement with the supplied experiments while remaining numerically robust and physically consistent.


