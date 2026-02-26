# REFINE

REFINE (Redundancy-Exploiting Follow-up-Informed Nonlinear Enhancement) is an algorithm that learns an item-aligned nonlinear preprocessing map that uses longitudinal redundancy—via follow-up–informed supervision during training—to stabilize baseline questionnaire items while preserving their original meaning. It then applies an exactly linear decoder from these stabilized baseline items to future symptom vectors, yielding a Bayes-optimal predictor whose globally interpretable coefficient matrix provides transparent prognostic attribution across outcomes.

All code was tested in R version 4.3.1.

# Installation

> library(devtools)

> install_github("ericstrobl/REFINE")

> library(REFINE)

# Sample Run

> data = generate_synthetic_data(N = 200, d = 10, q = 2, T = 3) # generate synthetic data with N = 200 subjects, d = 10 questionnaire items, q = 2 nuisance covariates, and T = 3 followup time points

> fit  <- refine_fit(data$Y0, data$followups, z_idx = data$z_idx, ntree = 1000) # fit REFINE model

> pred <- refine_predict(fit, data$Y0) # predict with REFINE model

# refine_fit() Description

Inputs

* `Y0`: numeric baseline matrix (N × p) containing questionnaire items and (optionally) nuisance covariates as columns.

* `followups`: list of length T; each element is a list with

  * `Yt`: numeric follow-up item matrix at time t (n_t × d_t)

  * `idx0`: integer vector (length n_t) mapping rows of Yt to the corresponding subject rows in Y0

* `z_idx` (optional): integer indices of nuisance columns in Y0 to be used as extra predictors in the nonlinear preprocessor (all other columns are treated as questionnaire items).

* Additional tuning parameters (e.g., ntree, nodesize, mtry, ridge, pinv_tol, …) controlling the random forest stabilizer and linear calibration.

Output

* A fitted model object of class refine_model containing, for each time point t, the estimated reconstruction `B_t`, intercept `a_t`, linear decoder `beta_t`, and nonlinear preprocessor `h_t`, plus metadata needed to apply the same baseline column split at prediction time.


