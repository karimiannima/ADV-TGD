# Reporting biometric verification results

This checklist is for offline defensive evaluation on approved data. The
accompanying tool summarizes aggregate decisions from an already frozen
verifier. It does not load a recognizer, generate examples, or choose thresholds.

## Define the operating point

- Record the recognizer and weights revision, preprocessing, score direction,
  threshold, and tie rule before evaluating the test split.
- Use separate calibration and test sets. Verify identity separation and
  provenance independently of file hashes. State when identity labels are
  unavailable; distinct images alone do not establish identity independence.
- For each operating point, report calibration and test denominators, false
  accepts among impostor comparisons, and false rejects among genuine
  comparisons. Distinguish ordinary impostor comparisons from specially
  constructed inputs; do not pool them into one false-accept rate.
- An inherited cutoff is a fixed score threshold, not evidence that the same
  false-accept rate holds on a new dataset or recognizer. Zero observed errors
  do not establish zero risk.

## Make uncertainty and comparisons auditable

- Reusing identities or images creates dependent comparisons. Use an
  uncertainty analysis at the independently sampled unit or label rates as
  descriptive. Do not treat every pair or every training seed as independent.
- For matched conditions, preserve all paired cases, report exclusions and
  failures, and analyze paired differences. Separate quality metrics from
  verification metrics and state the dataset size for distributional metrics.
- For speed comparisons, separate training, preprocessing, inference, and
  postprocessing. Report hardware, precision, resolution, batch size, warmup,
  and elapsed-time distribution. A step count is not a measured speedup.
- Keep raw face data, identity labels, scores, and generated examples out of
  public PRs unless explicitly approved and permitted by the data terms.

## Aggregate-only example

Pass counts from a single prespecified test operating point:

```sh
python3 tools/verification_summary.py --false-accepts 1 --impostor-trials 500 \
  --false-rejects 4 --genuine-trials 200
```

The JSON includes numerators, denominators, rates, and empirical rate
resolution (one event divided by the denominator). For *independent binomial
trials only*, add `--independent-trials` to request pointwise 95% Wilson
intervals. The tool cannot establish independence or validate calibration.
The example counts are synthetic.
