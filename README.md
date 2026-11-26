# Widefield ODMR inference and review ranking

Selected code accompanying *Fast Parameter Inversion and Reliability Assessment
for Bias-Field-Free Widefield NV Magnetic Imaging*. The release provides a
trained 200-frequency model, a repeated-scan preprocessing and diagnostics
example, and a fixed-fraction task ranking example.

## Quick start

Use Python 3.13. The package versions in requirements.txt were verified on
Windows with CPU inference. From the repository root:

    python -m venv .venv
    # Windows PowerShell:
    .\.venv\Scripts\Activate.ps1
    # Linux or macOS:
    # source .venv/bin/activate
    python -m pip install -r requirements.txt
    python run_k200_inference.py
    python analyze_repeated_spectrum.py examples/synthetic_repeated_counts.csv
    python rank_review_tasks.py examples/precomputed_risk_scores.csv review_flags.csv
    python verify_examples.py

The inference example returns 9.959386 MHz for the splitting and 0.080951 MHz
for its predicted standard deviation on the tested setup. The generated
doublet has a true splitting of 10 MHz. The ranking example selects example_4
for review. Small numerical differences may occur on other platforms.

## Inference input and output

Pass a two-column CSV with the header frequency_ghz,fluorescence using
the --input option. Frequencies must be in ascending order and match the original
500-point grid (2.8200 to 2.9198 GHz in 0.0002 GHz steps) or the fixed 200-point
subset defined in sparse_fast.py. The code selects the matching frequency
points and standardizes the spectrum before prediction.

The command prints the splitting and its model-predicted standard deviation in
MHz. The Python function infer_spectra returns all ten estimates and predicted
standard deviations in the order defined by PARAMETER_NAMES. The center is in
GHz, the splitting and half widths at half maximum are in MHz, and contrast
and mixing coefficients are dimensionless. The standard deviation is a network
output; the example does not establish uncertainty calibration.

## From repeated camera counts to spectral diagnostics

    python analyze_repeated_spectrum.py examples/synthetic_repeated_counts.csv
    python analyze_repeated_spectrum.py examples/measured_spiral_repeated_counts.csv --prepared-csv prepared_spectrum.csv

Both tables have 500 rows in ascending frequency order. Each row holds
frequency_ghz and the raw count at that frequency for each of ten complete
sweeps (repeat_01 through repeat_10). The measured example is one sampled
pixel at zero applied current from the separate planar spiral antenna
experiment. The CSV contains the per-sweep counts extracted from the camera
record, before averaging or standardization. The synthetic example uses
simulated Poisson draws with seed 441 and 160 times the relative fluorescence
values in synthetic_odmr_spectrum.csv as the mean counts.

Preprocessing averages the sweeps per frequency, computes the standard error
of the mean (SEM) from the repeats, and floors it at the larger of 1% of the
spectrum's median SEM and 1e-9 counts. The optional output contains all
500 frequencies with mean counts and SEM. The inference code uses the
standardized 200-point subset. The diagnostic fits an affine count scale
and offset to the predicted spectral shape and reports the root mean of
the squared noise-standardized residuals with two fitted degrees of freedom.
It also reports the splitting divided by the average half width at half maximum.

The measured example gives a splitting of about 11.652 MHz, a resolvability of
1.117, and a forward residual of 1.188. The release recomputes the SEM directly
from per-sweep counts. Archived results may use stored repeat standard
deviations and differ slightly. These outputs are single-spectrum diagnostics.
The task ranking example below starts from a separate table of precomputed
task-level risk scores.

## Task ranking input and output

Use a CSV with one row per task and frequency configuration, with these columns:

| Column | Meaning |
| --- | --- |
| k | Number of input frequencies |
| triplet_id | Task identifier |
| risk_residual | Calibrated forward-residual risk |
| risk_domain | Parameter-validity risk |
| risk_combined | Upstream combined risk score |
| risk_rho | Calibrated doublet-resolvability risk |
| raw_residual | Task forward residual before risk calibration |

The selector keeps the original priority order. It considers the priority flag
(residual risk or domain risk at least 1), combined risk, resolvability risk, raw
forward residual, and finally the string formed from the task ID and the
zero-based input row number. Larger keys receive priority. Within each k,
the number selected is 20% rounded to the nearest integer using Python's
rounding convention, with at least one task selected. Keep input row order to
preserve exact tie-breaking. The output contains k, triplet_id, and priority_review.

## Release scope

The package contains one trained 200-point model, synthetic examples, and one
measured spectrum exported as per-sweep counts. Camera image stacks, image
registration, parameter validity checks, score calibration, and multi-state task
aggregation are outside the release. Full benchmark reproduction also requires
additional model configurations, benchmark data, and reference fits. See the
manuscript for access to the remaining experimental data.


## Source provenance

The released copies of config.py, odmr_core.py, nn_model.py, pipeline.py, and
sparse_fast.py retain the original implementation. Release edits cover only
comments, docstrings, and blank lines. The original and release file hashes
are listed in source_sha256.txt. Training and other research
utilities present in these files are outside the supported examples.

| File | Provenance |
| --- | --- |
| submission_inference.py | Input adapter around the original grid, standardization, checkpoint loader, and prediction functions |
| preprocessing.py | Original cvf.py noise-floor function with a release CSV reader and repeat averaging adapter |
| spectral_diagnostics.py | Original recon_batch and weighted_affine_residual functions extracted from cvf.py and odmr_posthoc_reliability_audit.py with a dependency adapter |
| review_ranking.py | Original internal fixed-fraction selection functions from ptra_fig6_evidence_v2.py |
| Example and verification entry points | Added for the release |

The model export retains the trained tensors and architecture in the original
loader's format. Training state was omitted. Use the supplied weight file with
the example commands.
