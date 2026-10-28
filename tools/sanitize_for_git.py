"""Write sanitized copies of files that contain machine paths, for the Git-tracked part of Pitch-FlowNet.

Originals are never modified. For every candidate file (not ignored by the fixed rules in .gitignore)
whose text contains a machine path, this tool:
  - writes `<name>.sanitized<suffix>` beside it, with the paths replaced by placeholders;
  - lists the original in the generated block of .gitignore, so only the sanitized copy is tracked;
  - records both SHA-256 values in manifests/sanitization/<top-level folder>.json.
It also writes manifests/file-manifest-core.json: the entries of the local manifests/file-manifest.json
outside the dataset folders, with their Git status (tracked, sanitized copy, or local only).

Standard library only; nothing is trained, run or uploaded. Run from any directory:
    python tools/sanitize_for_git.py
"""
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Files that spell out the path patterns themselves (they name no account or machine).
SELF = {".gitignore", "tools/sanitize_for_git.py", "tools/check_tracked_package.py", "tests/test_tracked_package.py",
        "manifests/file-manifest-core.json"}   # the last one is written below from already-sanitized strings
BEGIN, END = "# BEGIN sanitized originals", "# END sanitized originals"
SEP = r"(?:\\\\|\\|/)+"          # one path separator: "/", "\" or a JSON-escaped "\\"
ONE = r"(?:\\\\|\\|/)"
USER = r"[^\\/\s\"'`]+"           # any Windows account name (the rules name no account)
WSL_UNC = r"(?:" + SEP + r"wsl(?:\.localhost|\$)" + SEP + r"[\w.-]+)?"
# Order matters: the longer Windows-workspace forms are replaced before the user-profile forms.
RULES = [
    ("<WINDOWS_WORKSPACE>", re.compile(r"/mnt/[a-z]/(?i:users)/" + USER + r"/OneDrive/Documents/Repository/FPT-Graduation-Project(?=/|\b)")),
    ("<WINDOWS_WORKSPACE>", re.compile(r"[A-Za-z]:" + SEP + "(?i:users)" + SEP + USER + SEP + "OneDrive" + SEP + "Documents" + SEP
                                       + "Repository" + SEP + r"FPT-Graduation-Project(?=\\|/|\b)")),
    ("<WINDOWS_USER_HOME>", re.compile(r"/mnt/[a-z]/(?i:users)/" + USER)),
    ("<WINDOWS_USER_HOME>", re.compile(r"[A-Za-z]:" + SEP + "(?i:users)" + SEP + USER)),
    ("<WSL_DEV_HOME>", re.compile(WSL_UNC + ONE + "home" + SEP + r"dev(?=\\|/|\b)")),
    ("<WSL_PROJECT_HOME>", re.compile(WSL_UNC + ONE + "home" + SEP + r"capstone(?=\\|/|\b)")),
    # A Windows path flattened into one directory name, e.g. "c--Users-<account>-OneDrive-...".
    ("<WINDOWS_WORKSPACE_SLUG>", re.compile(r"\b[A-Za-z]--(?i:users)-[A-Za-z0-9_.]+?-OneDrive-Documents-Repository-FPT-Graduation-Project")),
    ("<WINDOWS_USER_SLUG>", re.compile(r"\b[A-Za-z]--(?i:users)-[A-Za-z0-9_.]+")),
    # Default Windows host name of the workstation (it appears in TensorBoard event file names).
    ("<WORKSTATION>", re.compile(r"\bDESKTOP-[A-Z0-9]{7}\b")),
]
PLACEHOLDERS = {
    "<WSL_DEV_HOME>": "home directory of the WSL account that ran training and evaluation",
    "<WSL_PROJECT_HOME>": "home directory of a second WSL account on the same workstation",
    "<WINDOWS_WORKSPACE>": "the Windows project workspace (FPT-Graduation-Project)",
    "<WINDOWS_USER_HOME>": "the Windows user profile",
    "<WINDOWS_WORKSPACE_SLUG>": "the Windows project workspace path flattened into one directory name",
    "<WINDOWS_USER_SLUG>": "the Windows user profile path flattened into one directory name",
    "<WORKSTATION>": "host name of the training workstation",
}
# Anything that still matches after sanitizing is a defect.
RESIDUE = re.compile(r"/mnt/c/Users/|\b[A-Za-z]:(?:\\\\|\\|/)+Users(?:\\\\|\\|/)|(?:\\\\|\\|/)home(?:\\\\|\\|/)+(?:dev|capstone)\b"
                     r"|\b[A-Za-z]--Users-|\bDESKTOP-[A-Z0-9]{7}\b", re.IGNORECASE)


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def decode(raw: bytes):
    """Return (text, encoding) or (None, None) for binary content."""
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16"), "utf-16"
    if b"\0" in raw[:8192]:
        return None, None
    try:
        return raw.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        return None, None


def sanitize_text(text: str):
    counts = {}
    for name, rx in RULES:
        text, n = rx.subn(name, text)
        if n:
            counts[name] = counts.get(name, 0) + n
    return text, counts


def sanitized_name(rel: str) -> str:
    p = Path(rel)
    name = p.name
    stem, dot, suffix = name.rpartition(".")
    if not stem:                                    # ".bashrc" or "Makefile": no suffix to keep
        new = name + ".sanitized"
    else:
        new = f"{stem}.sanitized.{suffix}"
    return (p.parent / new).as_posix()


def gi_escape(rel: str) -> str:
    """Escape a literal path for .gitignore."""
    out = "".join("\\" + c if c in "*?[]\\" else c for c in rel)
    return out[:-1] + "\\ " if out.endswith(" ") else out


def list_files(exclude_text: str):
    with tempfile.NamedTemporaryFile("w", suffix=".gitignore", delete=False, encoding="utf-8") as f:
        f.write(exclude_text)
        tmp = f.name
    try:
        local = ROOT / ".git/info/exclude"      # per-clone excludes, never tracked
        extra = [f"--exclude-from={local}"] if local.is_file() else []
        out = subprocess.run(["git", "ls-files", "-z", "--cached", "--others", f"--exclude-from={tmp}", *extra],
                             cwd=ROOT, capture_output=True, check=True).stdout
    finally:
        Path(tmp).unlink()
    return sorted({p for p in out.decode("utf-8").split("\0") if p})


def main():
    gi_path = ROOT / ".gitignore"
    gi = gi_path.read_text(encoding="utf-8")
    head, rest = gi.split(BEGIN, 1)
    _, tail = rest.split(END, 1)
    fixed_rules = head + tail                       # .gitignore without the generated block

    mdir = ROOT / "manifests/sanitization"
    previous = {e["sanitized"] for m in mdir.glob("*.json") for e in json.loads(m.read_text(encoding="utf-8"))["files"]}
    maps, originals, problems = {}, [], []
    for rel in list_files(fixed_rules):
        if ".sanitized" in Path(rel).name or rel.startswith("manifests/sanitization/") or rel in SELF:
            continue
        path = ROOT / rel
        if not path.is_file():
            continue
        raw = path.read_bytes()
        text, enc = decode(raw)
        if text is None:
            if RESIDUE.search(raw.decode("latin-1")):
                problems.append(f"binary file with a machine path: {rel}")
            continue
        clean, counts = sanitize_text(text)
        if not counts:
            if RESIDUE.search(text):
                problems.append(f"unhandled machine path: {rel}")
            continue
        if RESIDUE.search(clean):
            problems.append(f"residue after sanitizing: {rel}")
            continue
        out = clean.encode(enc)
        dest_rel = sanitized_name(rel)
        dest = ROOT / dest_rel
        if not dest.exists() or dest.read_bytes() != out:
            dest.write_bytes(out)
        originals.append(rel)
        maps.setdefault(rel.split("/", 1)[0], []).append({
            "original": rel, "original_sha256": sha(raw), "original_bytes": len(raw),
            "sanitized": dest_rel, "sanitized_sha256": sha(out), "sanitized_bytes": len(out),
            "encoding": enc, "replacements": counts})

    if problems:
        sys.exit("\n".join(problems))

    gi_path.write_text(head + BEGIN + "\n" + "".join(f"/{gi_escape(r)}\n" for r in originals) + END + tail,
                       encoding="utf-8", newline="\n")
    current = {e["sanitized"] for entries in maps.values() for e in entries}
    for stale in sorted(previous - current):        # copies written earlier whose original no longer needs one
        if (ROOT / stale).is_file():
            (ROOT / stale).unlink()
    mdir.mkdir(parents=True, exist_ok=True)
    for old in mdir.glob("*.json"):
        if old.stem not in maps:
            old.unlink()
    for area, entries in sorted(maps.items()):
        doc = {"tool": "tools/sanitize_for_git.py", "placeholders": PLACEHOLDERS,
               "note": "The original stays on the workstation (git-ignored, unchanged). The sanitized copy differs only by the "
                       "placeholder replacements counted per file.", "files": entries}
        (mdir / f"{area}.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")

    write_core_manifest(set(originals))
    print(json.dumps({"sanitized_copies": len(originals), "by_area": {k: len(v) for k, v in sorted(maps.items())}}, indent=1))


def write_core_manifest(originals):
    full = ROOT / "manifests/file-manifest.json"
    if not full.exists():
        print("manifests/file-manifest.json not present (clone without local data); core manifest left as is")
        return
    tracked = set(list_files((ROOT / ".gitignore").read_text(encoding="utf-8")))
    entries = []
    for e in json.loads(full.read_text(encoding="utf-8"))["entries"]:
        d = e["destination"]
        if d.startswith(("datasets/raw/", "datasets/processed/")) or (d.startswith("history/")
                                                                    and not d.startswith("history/environment/")):
            continue
        e = {k: (sanitize_text(v)[0] if isinstance(v, str) else v) for k, v in e.items()}
        if d in originals:
            e["git"] = {"status": "sanitized_copy", "path": sanitized_name(d)}
        elif d in tracked:
            e["git"] = {"status": "tracked"}
        else:
            e["git"] = {"status": "local_only"}
        entries.append(e)
    doc = {"source": "manifests/file-manifest.json (local only; it also lists every dataset file)",
           "excluded": "datasets/raw/ and datasets/processed/ (summarised in manifests/dataset-manifest.json); "
                       "history/ apart from history/environment/ (kept on the workstation)",
           "entries": entries}
    (ROOT / "manifests/file-manifest-core.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n",
                                                           encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
