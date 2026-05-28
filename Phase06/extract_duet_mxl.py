"""
extract_duet_mxl.py — Selectively extract duet MXL files from PDMX archive

Streams through .cache/mxl.tar.gz and extracts only the MXL files for the
1,890 duet scores identified in Phase 6.1 (tracks containing both program 0
and program 40 in PDMX.csv).

Avoids the ~6 GB full-extraction cost; only the duet subset (~tens of MB)
is written to Phase06/mxl_duet/.
"""

import os
import csv
import tarfile
import sys

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
CSV_PATH    = os.path.join(PROJECT_DIR, "PDMX.csv")
TAR_PATH    = os.path.join(PROJECT_DIR, ".cache", "mxl.tar.gz")
OUT_DIR     = os.path.join(SCRIPT_DIR, "mxl_duet")


def collect_duet_mxl_paths() -> set[str]:
    """Read PDMX.csv and return the set of mxl tar entry names for duet scores."""
    wanted = set()
    with open(CSV_PATH, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            tracks = (row.get('tracks') or '').strip()
            if not tracks:
                continue
            parts = tracks.split('-')
            if '0' in parts and '40' in parts:
                # CSV mxl paths look like './mxl/8/31/Qm....mxl'
                # Tar member names look like 'mxl/8/31/Qm....mxl' (no ./)
                rel = (row.get('mxl') or '').strip().lstrip('./')
                if rel:
                    wanted.add(rel)
    return wanted


def main():
    if not os.path.exists(CSV_PATH):
        print(f"ERROR: {CSV_PATH} not found.")
        sys.exit(1)
    if not os.path.exists(TAR_PATH):
        print(f"ERROR: {TAR_PATH} not found. Download mxl.tar.gz into .cache/ first.")
        sys.exit(1)

    print(f"Reading {CSV_PATH} ...")
    wanted = collect_duet_mxl_paths()
    print(f"Duet scores to extract: {len(wanted):,}")

    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Streaming {TAR_PATH} ...")
    extracted = 0
    skipped   = 0
    missing   = set(wanted)   # paths we still need to find
    total_bytes = 0

    with tarfile.open(TAR_PATH, 'r:gz') as tar:
        for member in tar:
            if not member.isfile():
                continue
            if member.name not in wanted:
                continue

            # Extract to OUT_DIR/<rest after 'mxl/'>: preserve X/XX/Qm... layout
            rel_path = member.name[len('mxl/'):] if member.name.startswith('mxl/') else member.name
            out_path = os.path.join(OUT_DIR, rel_path)

            if os.path.exists(out_path) and os.path.getsize(out_path) == member.size:
                skipped += 1
                missing.discard(member.name)
                continue

            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            f = tar.extractfile(member)
            if f is None:
                continue
            with open(out_path, 'wb') as out:
                out.write(f.read())

            extracted += 1
            total_bytes += member.size
            missing.discard(member.name)

            if extracted % 100 == 0:
                print(f"  extracted {extracted:>5,} / {len(wanted):,}  "
                      f"({total_bytes / 1e6:.1f} MB so far)")

    print(f"\nDone.")
    print(f"  Extracted    : {extracted:,} files ({total_bytes / 1e6:.1f} MB)")
    print(f"  Already had  : {skipped:,} files")
    print(f"  Not found    : {len(missing):,} (paths in CSV but not in tarball)")
    print(f"  Output dir   : {OUT_DIR}")

    if missing:
        sample = list(missing)[:5]
        print(f"\n  Examples of missing paths:")
        for s in sample:
            print(f"    {s}")


if __name__ == '__main__':
    main()
