"""Summarize aggregate verifier decisions at a prespecified operating point."""

import argparse
import json
import math


def rate_summary(errors, trials, *, independent_trials=False):
    """Return a descriptive rate, with an opt-in binomial Wilson interval."""
    if type(errors) is not int or type(trials) is not int:
        raise ValueError("counts must be integers")
    if trials <= 0 or not 0 <= errors <= trials:
        raise ValueError("require trials > 0 and 0 <= errors <= trials")
    rate = errors / trials
    report = {
        "errors": errors,
        "trials": trials,
        "rate": rate,
        "rate_resolution": 1 / trials,
    }
    if independent_trials:
        z = 1.959963984540054
        z2 = z * z
        denominator = 1 + z2 / trials
        center = (rate + z2 / (2 * trials)) / denominator
        variance_term = (rate * (1 - rate) + z2 / (4 * trials)) / trials
        half_width = z * math.sqrt(variance_term) / denominator
        report["wilson_95"] = [
            0.0 if errors == 0 else max(0.0, center - half_width),
            1.0 if errors == trials else min(1.0, center + half_width),
        ]
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--false-accepts", type=int, required=True)
    parser.add_argument("--impostor-trials", type=int, required=True)
    parser.add_argument("--false-rejects", type=int, required=True)
    parser.add_argument("--genuine-trials", type=int, required=True)
    parser.add_argument("--independent-trials", action="store_true",
                        help="opt in to Wilson intervals only for independent binomial trials")
    args = parser.parse_args(argv)
    try:
        summary = {
            "false_accepts": rate_summary(args.false_accepts, args.impostor_trials,
                                          independent_trials=args.independent_trials),
            "false_rejects": rate_summary(args.false_rejects, args.genuine_trials,
                                          independent_trials=args.independent_trials),
        }
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
