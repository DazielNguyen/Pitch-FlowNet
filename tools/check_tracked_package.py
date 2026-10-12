"""Check the Git-tracked part of Pitch-FlowNet (read-only; standard library, PyYAML optional).

Works in two settings:
  - on the workstation, where the git-ignored originals and data are present;
  - in a clean clone, where only tracked files exist (checks that need originals are reported as skipped).

The checked set is what Git tracks or would track: `git ls-files --cached --others --exclude-standard`.

    python tools/check_tracked_package.py [--output report.json]
"""
import argparse
import datetime
import hashlib
import importlib.util
import json
import platform
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("sanitize_for_git", ROOT / "tools/sanitize_for_git.py")
san = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(san)

MAX_BYTES = 5_000_000
FORBIDDEN_SUFFIX = {".ckpt", ".pt", ".pth", ".onnx", ".wav", ".so", ".pyd", ".pyc", ".bundle", ".pack"}
FORBIDDEN_PREFIX = ("datasets/raw/", "datasets/processed/", "history/private/", "evidence/manuscript/", "history/notes/")
FORBIDDEN_NAME = re.compile(r"^events\.out\.tfevents\.|^\.env(\..*)?$|\.(key|pem)$")
# Upstream authors named in third-party code (setup.py of Piper, nlohmann/json); any other address is reported.
EMAIL_ALLOW = {"mike@rhasspy.org", "bjoern@hoehrmann.de", "evan@nemerson.com"}
EMAIL = re.compile(r"\b[\w.+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}\b")
SECRET = re.compile(r"\b(?:hf_[A-Za-z0-9]{30,}|gh[pousr]_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|xox[baprs]-[A-Za-z0-9-]{10,})\b")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def git(*args) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, check=True).stdout.decode("utf-8")


def text_of(p: Path):
    raw = p.read_bytes()
    text, _ = san.decode(raw)
    return text


class Report:
    def __init__(self):
        self.checks = []

    def add(self, name, ok, detail=None, skipped=False):
        self.checks.append({"check": name, "result": "skipped" if skipped else ("pass" if ok else "FAIL"),
                            **({"detail": detail} if detail else {})})

    @property
    def failed(self):
        return [c["check"] for c in self.checks if c["result"] == "FAIL"]


def run():
    r = Report()
    files = sorted({f for f in git("ls-files", "-z", "--cached", "--others", "--exclude-standard").split("\0") if f})
    files = [f for f in files if (ROOT / f).is_file()]
    fileset = set(files)

    # 1. No artifacts, private folders, oversized or credential files.
    bad = [f for f in files if Path(f).suffix.lower() in FORBIDDEN_SUFFIX or f.startswith(FORBIDDEN_PREFIX)
           or FORBIDDEN_NAME.search(Path(f).name)]
    r.add("no datasets, checkpoints, audio, events, compiled files or private folders", not bad, bad[:20])
    big = [f"{f} ({(ROOT / f).stat().st_size} bytes)" for f in files if (ROOT / f).stat().st_size > MAX_BYTES]
    r.add(f"no file over {MAX_BYTES} bytes", not big, big)

    # 2. No machine paths, host names, personal e-mail addresses or tokens in tracked text.
    residue, emails, secrets = [], [], []
    for f in files:
        if f in san.SELF - {"manifests/file-manifest-core.json"}:
            continue
        t = text_of(ROOT / f)
        if t is None:
            continue
        if san.RESIDUE.search(t):
            residue.append(f)
        found = {e for e in EMAIL.findall(t)} - EMAIL_ALLOW
        if found:
            emails.append(f)
        if SECRET.search(t):
            secrets.append(f)
    r.add("no machine paths or host names in tracked text", not residue, residue[:20])
    r.add("no e-mail addresses other than upstream authors", not emails, emails[:20])
    r.add("no token-like strings", not secrets, secrets[:20])

    # 3. Sanitized copies match their maps; originals (when present) match and stay ignored.
    entries = []
    for m in sorted((ROOT / "manifests/sanitization").glob("*.json")):
        entries += json.loads(m.read_text(encoding="utf-8"))["files"]
    mapped = {e["sanitized"] for e in entries}
    wrong, rederive, not_ignored, originals_present = [], [], [], 0
    for e in entries:
        s = ROOT / e["sanitized"]
        if e["sanitized"] not in fileset or sha(s) != e["sanitized_sha256"]:
            wrong.append(e["sanitized"])
        o = ROOT / e["original"]
        if o.is_file():
            originals_present += 1
            raw = o.read_bytes()
            text, enc = san.decode(raw)
            if sha(o) != e["original_sha256"] or san.sanitize_text(text)[0].encode(enc) != s.read_bytes():
                rederive.append(e["original"])
            if e["original"] in fileset:
                not_ignored.append(e["original"])
    stray = [f for f in files if ".sanitized" in Path(f).name and f not in mapped]
    r.add("every sanitized copy is mapped and matches its SHA-256", not wrong and not stray, (wrong + stray)[:20])
    if originals_present:
        r.add("originals match the map and re-sanitize to the tracked copy", not rederive,
              {"originals_present": originals_present, "mismatch": rederive[:20]})
        r.add("originals with machine paths are git-ignored", not not_ignored, not_ignored[:20])
    else:
        r.add("originals match the map and re-sanitize to the tracked copy", True, "originals not present (clone)", skipped=True)

    # 4. Tracked copies of recorded files still carry their recorded bytes.
    core = json.loads((ROOT / "manifests/file-manifest-core.json").read_text(encoding="utf-8"))["entries"]
    changed, missing, leaked = [], [], []
    for e in core:
        status, d = e["git"]["status"], e["destination"]
        if status == "tracked":
            if d not in fileset:
                missing.append(d)
            elif sha(ROOT / d) != e["destination_sha256"]:
                changed.append(d)
        elif status == "sanitized_copy":
            if e["git"]["path"] not in fileset:
                missing.append(e["git"]["path"])
            if d in fileset:
                leaked.append(d)
        elif d in fileset:
            leaked.append(d)
    r.add("recorded files tracked as-is keep their recorded SHA-256", not changed and not missing,
          {"changed": changed[:20], "missing": missing[:20], "entries": len(core)})
    r.add("local-only and sanitized originals are not tracked", not leaked, leaked[:20])

    # 5. Structured files parse.
    bad_json = []
    for f in files:
        if f.endswith((".json", ".jsonl")):
            t = text_of(ROOT / f)
            try:
                if f.endswith(".jsonl"):
                    [json.loads(line) for line in t.splitlines() if line.strip()]
                else:
                    json.loads(t)
            except (ValueError, TypeError) as exc:
                bad_json.append(f"{f}: {exc}")
    r.add("JSON and JSONL files parse", not bad_json, bad_json[:20])
    try:
        import yaml
    except ImportError:
        r.add("YAML files parse", True, "PyYAML not installed", skipped=True)
    else:
        # Lightning hparams carry python/* tags (tuples, PosixPath). Read them as plain data; never build the objects.
        class TagTolerantLoader(yaml.SafeLoader):
            pass

        def plain(loader, suffix, node):
            if isinstance(node, yaml.MappingNode):
                return loader.construct_mapping(node, deep=True)
            if isinstance(node, yaml.SequenceNode):
                return loader.construct_sequence(node, deep=True)
            return loader.construct_scalar(node)

        TagTolerantLoader.add_multi_constructor("tag:yaml.org,2002:python/", plain)
        bad_yaml = []
        for f in files:
            if f.endswith((".yaml", ".yml")):
                try:
                    yaml.load(text_of(ROOT / f), Loader=TagTolerantLoader)
                except yaml.YAMLError as exc:
                    bad_yaml.append(f"{f}: {str(exc).splitlines()[0]}")
        r.add("YAML files parse", not bad_yaml, bad_yaml[:20])

    # 6. Byte-exact checkout.
    attrs = (ROOT / ".gitattributes").read_text(encoding="utf-8") if (ROOT / ".gitattributes").is_file() else ""
    r.add(".gitattributes disables end-of-line conversion", "* -text" in attrs.splitlines())

    # 7. Local Markdown links resolve to tracked files or folders.
    broken = []
    for f in files:
        if f.endswith(".md") and not f.startswith("code/"):
            for target in re.findall(r"\]\(([^)#\s]+)(?:#[^)]*)?\)", text_of(ROOT / f)):
                if re.match(r"[a-z]+:", target):
                    continue
                path = (ROOT / f).parent / target
                rel = path.resolve().relative_to(ROOT.resolve()).as_posix() if path.resolve().is_relative_to(ROOT.resolve()) else None
                if rel is None or not (rel in fileset or any(x.startswith(rel.rstrip("/") + "/") for x in fileset)):
                    broken.append(f"{f} -> {target}")
    r.add("relative Markdown links in project documents resolve", not broken, broken[:20])

    return r, files


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--output", help="write the JSON report here")
    args = ap.parse_args()
    r, files = run()
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    report = {"tool": "tools/check_tracked_package.py",
              "run_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
              "python": platform.python_version(), "git_head": head or None, "files_checked": len(files),
              "checks": r.checks, "failed": r.failed}
    text = json.dumps(report, indent=1) + "\n"
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8", newline="\n")
    print(text, end="")
    sys.exit(1 if r.failed else 0)


if __name__ == "__main__":
    main()
