# Thermal Interface Identification

You are given a Python project for investigating transient heat transfer through a wall made from two different materials.

The project contains three experimental datasets:

- `data/heating.csv`
- `data/cooling.csv`
- `data/moderate_heating.csv`

The existing implementation in `src/` models the wall and calibrates thermal conductivities for the two materials. The current model does not reproduce the experimental measurements accurately.

## Objective

Investigate the existing physical model, numerical implementation, experimental data, and calibration procedure. Determine why the current model fails to reproduce the measurements and implement an improved physical and numerical model that explains the observed temperature behavior.

The improved model must use physically meaningful parameters rather than hardcoded corrections or dataset-specific transformations.

## Required work

1. Inspect the existing source code and all three experimental datasets.

2. Run the existing implementation and quantify where and how its predictions disagree with the measurements.

3. Use the observed discrepancies to determine an appropriate physical improvement to the model. The improvement should be supported by the experimental behavior rather than chosen arbitrarily.

4. Extend the parameter estimation procedure so that all required physical parameters introduced by the improved model can be calibrated from the provided experiments.

5. Ensure calibrated physical parameters are finite, positive, reproducible, and stable across reasonable initial parameter guesses.

6. Validate the improved model against all three visible experiments using quantitative error metrics.

7. Check that the numerical solver remains stable and produces finite, physically sensible temperatures under the provided experiments and reasonable parameter values.

8. Demonstrate that the calibrated model can make predictions for an operating condition that is not present in the visible datasets. The evaluation environment contains additional validation data that must not be used during calibration.

9. Create a scientific report at `report.md` describing:
   - the original model and its limitations
   - evidence found in the experimental data
   - the physical interpretation of the improvement
   - the calibrated parameters and their units
   - the calibration procedure
   - the numerical method and stability considerations
   - quantitative validation results for the visible experiments
   - the approach used for unseen-condition prediction
   - reproducibility considerations

## Constraints

- Use only the data and software available in the task environment.
- Do not use network services or external datasets.
- Do not hardcode expected parameter values, hidden results, or dataset-specific outputs.
- Do not access or attempt to reconstruct evaluator-only validation data.
- Preserve the existing dataset format.
- Preserve the existing Python APIs where practical so that the project remains usable by the supplied validation code.
- The final implementation must represent a genuine physical model rather than merely fitting, transforming, or memorizing the observed temperature values.
- Keep the implementation deterministic and reproducible.

## Completion criteria

The task is complete when:

- the physical model has been meaningfully improved;
- the required physical parameters can be calibrated from the visible experiments;
- calibration produces finite, positive, reproducible parameters;
- the improved model substantially reduces prediction error on the visible experiments;
- the model remains numerically stable and physically sensible;
- the calibrated model generalizes to the unseen evaluation condition;
- and `report.md` contains sufficient scientific and numerical documentation to reproduce and understand the investigation.