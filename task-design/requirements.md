# Thermal Interface Identification Task Requirements

## 1. Task Purpose

The task should require an agent to investigate and improve a transient
one-dimensional thermal model for a two-material layered wall.

The provided starter model assumes perfect thermal contact between the two
materials. Visible experimental data shows behavior that the starter model
cannot reproduce accurately.

The agent must diagnose the model-data mismatch, improve the physical model,
calibrate the unknown material/interface parameters, and demonstrate that the
result generalizes beyond the visible calibration experiments.

The task should require scientific reasoning, numerical modeling, parameter
estimation, debugging, validation, and creation of reproducible scientific
artifacts.

---

## 2. Agent Starting State

The agent receives:

- A Python project containing the starter thermal model.
- Three visible experimental datasets:
  - `heating`
  - `cooling`
  - `moderate_heating`
- Python dependencies required to run the project.
- Existing analysis scripts that allow the agent to inspect the data and
  diagnose the starter model.

The starter model currently estimates only:

- thermal conductivity of material A
- thermal conductivity of material B

The starter model assumes perfect thermal contact at the material interface.

The agent should not be given the true parameter values or hidden validation
datasets.

---

## 3. Scientific Problem

The wall consists of two different materials arranged in series.

The model represents the wall using a one-dimensional finite-volume
discretization and predicts transient temperature evolution.

Temperature measurements are available at locations inside both materials
and on opposite sides of their interface.

The visible experiments demonstrate that the starter model cannot reproduce
the measured temperature behavior, particularly the persistent temperature
difference across the material interface during heating experiments.

The agent must determine an appropriate physical and numerical explanation
for the mismatch and implement a model capable of reproducing the observed
behavior.

The task instruction must not directly state the missing physical mechanism
or provide its implementation formula.

---

## 4. Required Agent Work

A successful solution should require the agent to:

1. Inspect the existing model and experimental data.
2. Run the starter model and establish its failure.
3. Investigate the source of the mismatch.
4. Modify the physical model appropriately.
5. Extend parameter calibration to account for the improved model.
6. Validate the resulting model against the visible experiments.
7. Test the model under conditions not used for calibration.
8. Produce reproducible numerical results and scientific artifacts.

The task should permit different valid implementation strategies.

The grader should evaluate observable scientific behavior rather than requiring
one specific source-code implementation.

---

## 5. Calibration Requirements

The final model must estimate the unknown physical parameters from the visible
experimental datasets.

Calibration should:

- maintain physically valid parameter values;
- use multiple visible experiments;
- produce reproducible fitted parameters;
- achieve substantially better agreement with the measured data than the
  starter model;
- avoid relying on hard-coded answers.

The grader should verify calibration using independent calculations where
possible.

---

## 6. Validation Requirements

Validation must be separated from calibration.

The final solution should be evaluated on conditions that are not directly
used to fit the parameters.

Hidden validation should include at least one experiment with different
boundary and/or initial conditions.

The hidden validation data must not be exposed to the agent.

The final model should reproduce hidden temperature measurements within
scientifically justified numerical tolerances.

Validation should use more than one numerical metric where useful, such as:

- RMSE
- maximum absolute error
- sensor-specific errors

No single metric should be the sole basis for determining whether the
scientific model is correct.

---

## 7. Physical Consistency

The final model must preserve physically meaningful behavior.

The grader should check properties such as:

- positive thermal conductivities;
- positive interface-related physical parameters where applicable;
- temperatures remaining numerically finite;
- stable numerical integration;
- sensible response to changes in boundary conditions;
- correct direction of heat transfer;
- consistency between predicted interface behavior and the experimental data.

Tests should focus on observable physical behavior rather than requiring
specific variable names or implementation details.

---

## 8. Robustness

The final implementation should work from reasonable parameter initial
guesses rather than depending on one carefully selected starting point.

The solution should also:

- handle the provided datasets consistently;
- avoid non-finite numerical results;
- fail clearly for invalid physical parameters;
- produce deterministic results;
- remain reproducible when executed in the supplied environment.

Where appropriate, hidden tests should perturb initial guesses or evaluate
additional valid conditions.

---

## 9. Required Scientific Artifacts

The completed solution should produce machine-readable and human-readable
results.

Expected artifacts include:

- calibrated parameter values;
- validation metrics;
- a calibration/validation summary;
- generated plots or equivalent numerical evidence showing agreement between
  model and measurements.

Artifacts must be generated from the model execution rather than manually
entered.

The exact artifact filenames and schemas will be defined by the final task
package and grader.

---

## 10. Grading Dimensions

The final benchmark should contain independent grading dimensions covering
different capabilities.

Candidate dimensions:

### A. Physical Model Improvement

The modified model must reproduce the interface behavior that the starter
model cannot explain.

### B. Parameter Calibration

The unknown physical parameters must be estimated from the visible data and
remain physically valid.

### C. Visible Experiment Validation

The calibrated model must provide substantially improved predictions across
the visible experiments.

### D. Hidden Generalization

The same calibrated model must predict an unseen experimental condition
without access to its measurements during calibration.

### E. Numerical Robustness

The implementation must remain stable, finite, reproducible, and usable from
multiple reasonable starting conditions.

### F. Scientific Reporting

The required calibration and validation artifacts must be generated and
contain values consistent with the actual model execution.

Each dimension should test a genuinely different capability and should avoid
simply repeating the same RMSE threshold.

---

## 11. Hidden Information

The following information should remain unavailable to the agent:

- true physical parameter values;
- hidden experiment definitions;
- hidden measurements;
- hidden expected numerical results;
- grader implementation details;
- oracle solution details.

The hidden data should be stored inside the evaluation environment and accessed
only by the tests.

---

## 12. Determinism

The task must be reproducible.

The final Docker environment should pin dependency ranges or versions
sufficiently to avoid meaningful numerical differences.

Synthetic data generation, if used during task construction, must use fixed
random seeds.

The grader should use deterministic tolerances rather than exact floating
point equality.

---

## 13. Design Constraints

The final task must not:

- require internet access;
- depend on external services;
- require access to proprietary APIs;
- reveal the intended physical fix in the task instruction;
- depend on a single library call;
- be solvable by simply copying a hard-coded parameter set;
- use a generic software-engineering ticket with a scientific wrapper.

The scientific problem itself should create the difficulty.

The agent should need to inspect multiple files, run experiments, reason about
the numerical model, iterate on the implementation, and validate the result.

---

## 14. Success Definition

A successful agent solution is one that:

1. identifies the limitation of the starter model through evidence;
2. implements a physically meaningful improvement;
3. calibrates the resulting model using the visible experiments;
4. substantially improves agreement with the visible measurements;
5. generalizes to hidden experimental conditions;
6. remains numerically robust;
7. produces reproducible scientific artifacts.

The grader must verify these properties independently of the agent's
explanation.