import json
from pathlib import Path
import subprocess
import sys
import unittest

from tools.verification_summary import rate_summary


class VerificationSummaryTests(unittest.TestCase):
    def test_default_is_descriptive(self):
        self.assertEqual(rate_summary(1, 500), {
            "errors": 1, "trials": 500, "rate": 0.002, "rate_resolution": 0.002,
        })

    def test_zero_errors_does_not_mean_zero_upper_bound(self):
        lower, upper = rate_summary(0, 100, independent_trials=True)["wilson_95"]
        self.assertEqual(lower, 0.0)
        self.assertAlmostEqual(upper, 0.0369934982)

    def test_symmetric_interval(self):
        lower, upper = rate_summary(50, 100, independent_trials=True)["wilson_95"]
        self.assertAlmostEqual(lower, 0.4038315304)
        self.assertAlmostEqual(upper, 1 - lower)

    def test_all_errors(self):
        lower, upper = rate_summary(100, 100, independent_trials=True)["wilson_95"]
        self.assertAlmostEqual(lower, 0.9630065018)
        self.assertEqual(upper, 1.0)

    def test_invalid_counts(self):
        for errors, trials in ((0, 0), (2, 1), (-1, 1), (1, -1), (0.5, 10), (True, 1)):
            with self.subTest(errors=errors, trials=trials), self.assertRaises(ValueError):
                rate_summary(errors, trials)

    def test_cli(self):
        script = Path(__file__).resolve().parents[1] / "tools" / "verification_summary.py"
        command = [sys.executable, "-B", str(script), "--false-accepts", "1",
                   "--impostor-trials", "500", "--false-rejects", "4",
                   "--genuine-trials", "200"]
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        self.assertEqual(json.loads(result.stdout)["false_rejects"]["rate"], 0.02)
        bad = subprocess.run(command + ["--genuine-trials", "0"],
                             capture_output=True, text=True)
        self.assertEqual(bad.returncode, 2)
        self.assertEqual(bad.stdout, "")
        self.assertIn("require trials > 0", bad.stderr)


if __name__ == "__main__":
    unittest.main()
