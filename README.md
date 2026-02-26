# REFINE

REFINE (Redundancy-Exploiting Follow-up-Informed Nonlinear Enhancement) is an algorithm that learns an item-aligned nonlinear preprocessing map that uses longitudinal redundancy—via follow-up–informed supervision during training—to stabilize baseline questionnaire items while preserving their original meaning. It then applies an exactly linear decoder from these stabilized baseline items to future symptom vectors, yielding a Bayes-optimal predictor whose globally interpretable coefficient matrix provides transparent prognostic attribution across outcomes.

All code was tested in R version 4.3.1.

# Installation

> library(devtools)

> install_github("ericstrobl/REFINE")

> library(REFINE)

# Sample Run

