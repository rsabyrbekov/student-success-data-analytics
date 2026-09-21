# Deployment boundaries

This repository has no production deployment target. It provides offline
training and report generation only. Do not expose the baseline as an
unreviewed API, use it for high-impact decisions, or connect it to student
records without institutional approval and privacy/security review.

Any future service must enforce authentication, authorization, encryption,
audit logging, input validation, rate limits, secret management, version
pinning, monitoring, rollback, and a human-review gate. Promotion requires
passing data-quality and fairness gates on representative validation data; a
failed gate must block release.
