"""Tests for the sanitizing rules and the tracked-package checks (standard library unittest).

    python -m unittest discover -s tests -v
"""
import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


san = load("sanitize_for_git")
check = load("check_tracked_package")


class SanitizeRules(unittest.TestCase):
    # "alice" stands in for any account name; the rules must not depend on a specific one.
    CASES = {
        "/home/dev/data_preprocessed/dataset.jsonl": "<WSL_DEV_HOME>/data_preprocessed/dataset.jsonl",
        "cd /home/dev && ls": "cd <WSL_DEV_HOME> && ls",
        "/home/capstone/handoff/x": "<WSL_PROJECT_HOME>/handoff/x",
        "/mnt/c/Users/alice/OneDrive/Documents/Repository/FPT-Graduation-Project/EdgeTTS/":
            "<WINDOWS_WORKSPACE>/EdgeTTS/",
        "/mnt/c/Users/alice/AppData/Local/Temp": "<WINDOWS_USER_HOME>/AppData/Local/Temp",
        "C:\\Users\\alice\\OneDrive\\Documents\\Repository\\FPT-Graduation-Project\\EdgeTTS":
            "<WINDOWS_WORKSPACE>\\EdgeTTS",
        '"C:\\\\Users\\\\alice\\\\anaconda3"': '"<WINDOWS_USER_HOME>\\\\anaconda3"',
        "c:/users/alice/x": "<WINDOWS_USER_HOME>/x",
        "\\\\wsl.localhost\\Ubuntu-24.04\\home\\dev\\runs": "<WSL_DEV_HOME>\\runs",
        "Temp/claude/c--Users-alice-OneDrive-Documents-Repository-FPT-Graduation-Project/abc":
            "Temp/claude/<WINDOWS_WORKSPACE_SLUG>/abc",
        "events.out.tfevents.1783117787.DESKTOP-ABC1234.5513.0": "events.out.tfevents.1783117787.<WORKSTATION>.5513.0",
    }

    def test_machine_paths_become_placeholders(self):
        for raw, expected in self.CASES.items():
            with self.subTest(raw=raw):
                out, counts = san.sanitize_text(raw)
                self.assertEqual(out, expected)
                self.assertTrue(counts)
                self.assertIsNone(san.RESIDUE.search(out))

    def test_unrelated_text_is_unchanged(self):
        for raw in ("/home/developer/x", "/usr/lib/python3", "users/alice", "DESKTOP-environment", "<WSL_DEV_HOME>/x"):
            with self.subTest(raw=raw):
                self.assertEqual(san.sanitize_text(raw), (raw, {}))

    def test_sanitized_names_keep_the_suffix(self):
        self.assertEqual(san.sanitized_name("runs/full/train_h.log"), "runs/full/train_h.sanitized.log")
        self.assertEqual(san.sanitized_name("code/x/.wsl_train.sh"), "code/x/.wsl_train.sanitized.sh")
        self.assertEqual(san.sanitized_name("code/x/Makefile"), "code/x/Makefile.sanitized")

    def test_gitignore_escaping(self):
        self.assertEqual(san.gi_escape("a/b[1]*.log"), "a/b\\[1\\]\\*.log")
        self.assertEqual(san.gi_escape("best-epoch=773-val_loss_mel=20.2158.csv"), "best-epoch=773-val_loss_mel=20.2158.csv")

    def test_rules_name_no_account_or_host(self):
        source = (ROOT / "tools/sanitize_for_git.py").read_text(encoding="utf-8")
        self.assertNotRegex(source, r"Users[\\/]+[a-z]{3,}[\\/]")
        self.assertNotRegex(source, r"DESKTOP-[A-Z0-9]{7}")


class TrackedPackage(unittest.TestCase):
    def test_all_checks_pass(self):
        report, files = check.run()
        self.assertTrue(files)
        self.assertEqual(report.failed, [], [c for c in report.checks if c["result"] == "FAIL"])


if __name__ == "__main__":
    unittest.main()
