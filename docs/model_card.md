# Model card: student-success baseline

This transparent logistic-regression example estimates persistence to the next
term from synthetic, non-identifying features. It is for education, testing,
and methodology review only. It is **not** approved for admissions, discipline,
eligibility, resource denial, or fully automated student decisions.

The data generator is deterministic and intentionally does not represent a
real institution. The 75/25 stratified split and seed are recorded in
`configs/baseline.json`. Data-quality checks reject missing, duplicate,
out-of-range, or single-class inputs. Reports include ROC AUC, threshold
metrics, Brier score, subgroup rates, and disparities.

Before any pilot, use a time-based, institution-specific validation set,
uncertainty intervals, calibration review, drift monitoring, and
intersectional analysis. Outputs may support voluntary outreach only, with
trained staff review, student notice, correction and appeal paths, access
controls, retention limits, and an equity review. See the governance and
deployment documents.
