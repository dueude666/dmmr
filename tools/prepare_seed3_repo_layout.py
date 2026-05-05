import argparse
from collections import defaultdict
from pathlib import Path
import re
import shutil


MAT_PATTERN = re.compile(r"^(\d+)_([0-9]+)\.mat$")


def ensure_parent(path):
    path.parent.mkdir(parents=True, exist_ok=True)


def link_or_copy(src, dst, mode):
    ensure_parent(dst)
    if dst.exists():
        return
    if mode in ("auto", "hardlink"):
        try:
            dst.hardlink_to(src)
            return
        except OSError:
            if mode == "hardlink":
                raise
    shutil.copy2(src, dst)


def collect_subject_files(raw_dir):
    subject_files = defaultdict(list)
    for path in sorted(raw_dir.glob("*.mat")):
        if path.name == "label.mat":
            continue
        match = MAT_PATTERN.match(path.name)
        if match is None:
            continue
        subject_id = int(match.group(1))
        session_key = match.group(2)
        subject_files[subject_id].append((session_key, path))
    return subject_files


def prepare_layout(raw_dir, output_root, mode):
    subject_files = collect_subject_files(raw_dir)
    if not subject_files:
        raise FileNotFoundError("No SEED .mat files were found in the raw directory.")

    output_root.mkdir(parents=True, exist_ok=True)
    label_path = raw_dir / "label.mat"
    if label_path.exists():
        link_or_copy(label_path, output_root / "label.mat", mode)
    readme_path = raw_dir / "readme.txt"
    if readme_path.exists():
        link_or_copy(readme_path, output_root / "readme.txt", mode)

    summary = []
    for subject_id in sorted(subject_files):
        files = sorted(subject_files[subject_id], key=lambda item: item[0])
        for session_index, (_, src_path) in enumerate(files, start=1):
            dst = output_root / str(session_index) / src_path.name
            link_or_copy(src_path, dst, mode)
        summary.append((subject_id, len(files)))
    return summary


def main():
    parser = argparse.ArgumentParser(description="Reorganize official SEED ExtractedFeatures into the repo layout.")
    parser.add_argument("--raw-dir", required=True, help="Flat official SEED ExtractedFeatures directory")
    parser.add_argument("--output-root", required=True, help="Output ExtractedFeatures root expected by the repo")
    parser.add_argument("--mode", choices=["auto", "hardlink", "copy"], default="auto", help="How to materialize files")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir).resolve()
    output_root = Path(args.output_root).resolve()
    summary = prepare_layout(raw_dir, output_root, args.mode)

    print("Prepared SEED layout at:", output_root)
    for subject_id, file_count in summary:
        print(f"subject {subject_id}: {file_count} session files")


if __name__ == "__main__":
    main()
