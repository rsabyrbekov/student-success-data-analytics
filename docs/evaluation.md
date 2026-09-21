# Evaluation and fairness checks

The baseline reports ROC AUC, balanced accuracy, accuracy, precision, recall,
F1, and Brier score. Brier score and calibration plots are important because
support teams need probabilities they can interpret, not only rankings.

Subgroup tables report sample size, observed positive rate, predicted positive
rate, and recall for `first_generation`, `financial_aid`, `online_student`, and
`age_band`, plus false-positive rate. Reports include demographic-parity,
equal-opportunity (recall), and false-positive-rate differences. These are
screening diagnostics, not proof that a model is fair. Always inspect
intersectional groups, confidence intervals, threshold effects, missingness,
and the consequences of false positives and false negatives.
