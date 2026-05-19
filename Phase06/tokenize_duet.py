"""
tokenize_duet.py — Per-track tokenization for piano+violin duet scores

For each MXL under Phase06/mxl_duet/, write a JSON to Phase06/tokens_duet/
containing per-track ST+ token streams:

    {
      "piano":   [ "bar", ..., "R", ..., "L", ..., "bar", ... ],
      "violin":  [ "bar", ..., "bar", ... ],
      "tracks":  "0-40-42",     # original CSV tracks string
      "n_bars":  104,
    }

Design decisions (per spec Phase 6.1 Option B):
  - Use midi-program (NOT part-name) to identify piano vs violin.
  - For multi-violin scores (e.g. 0-40-40), keep Violin 1 only (first
    program=40 part).
  - For multi-piano scores (rare), keep Piano 1 only.
  - Drop every other instrument (viola 41, cello 42, flute 73, ...) —
    do NOT merge them into the piano, since that would worsen the
    train/test distribution gap between synthesized pseudo-solos and
    real piano solos.

Pseudo-piano-solo synthesis (the reverse augmentation step) is deferred
to build_pairs_duet.py so we can iterate on the merging strategy without
re-tokenizing.
"""

import csv
import json
import os
import sys
import warnings
import zipfile
from collections import Counter

from bs4 import BeautifulSoup
from music21 import musicxml

# Silence music21's chatty XML warnings (the existing pipeline does this too)
warnings.filterwarnings("ignore", category=musicxml.xmlToM21.MusicXMLWarning)

# Force UTF-8 stdout so foreign-language part-names don't crash Windows console
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Reuse existing low-level helpers from score_to_tokens.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from score_to_tokens import (
    load_MusicXML, measure_to_tokens, common, others,
)

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
CSV_PATH    = os.path.join(PROJECT_DIR, "PDMX.csv")
MXL_DIR     = os.path.join(SCRIPT_DIR, "mxl_duet")
OUT_DIR     = os.path.join(SCRIPT_DIR, "tokens_duet")


# ───────────────────────── part / program inspection ─────────────────────────

def inspect_score_parts(mxl_path):
    """Parse <score-part>/<midi-instrument> metadata; return list of dicts
    in document order: [{id, program, staves}, ...]."""
    if mxl_path.lower().endswith('.mxl'):
        with zipfile.ZipFile(mxl_path, 'r') as z:
            xml_filename = next(
                (n for n in z.namelist()
                 if n.endswith('.xml') and n != 'META-INF/container.xml'),
                None,
            )
            with z.open(xml_filename) as f:
                soup = BeautifulSoup(f, 'lxml-xml')
    else:
        with open(mxl_path, encoding='utf-8') as f:
            soup = BeautifulSoup(f, 'lxml-xml')

    part_meta = {}
    for sp in soup.find_all('score-part'):
        pid = sp.get('id')
        prog = None
        mi = sp.find('midi-instrument')
        if mi:
            mp = mi.find('midi-program')
            if mp:
                try:
                    prog = int(mp.text) - 1   # MusicXML midi-program is 1-indexed
                except ValueError:
                    pass
        part_meta[pid] = {'id': pid, 'program': prog}

    result = []
    for part in soup.find_all('part'):
        pid = part.get('id')
        meta = part_meta.get(pid, {'id': pid, 'program': None}).copy()
        staves = {s.text for s in part.find_all('staff')}
        meta['staves'] = len(staves) if staves else 1
        result.append(meta)

    return result


# ───────────────────────── per-part tokenization ─────────────────────────────

def tokenize_single_part(measures, soup, staves, note_name=True):
    """Tokenize ONE part. Piano (staves==2) gets R/L split; violin (staves==1)
    is a single bar-major stream."""
    tokens = []
    if staves >= 2:
        for measure in measures:
            R = measure_to_tokens(measure, soup, 1, note_name)
            L = measure_to_tokens(measure, soup, 2, note_name)
            tokens += ['bar'] + common(R) + ['R'] + others(R) + ['L'] + others(L)
    else:
        for measure in measures:
            M = measure_to_tokens(measure, soup, None, note_name)
            tokens += ['bar'] + M
    return tokens


def tokenize_duet_mxl(mxl_path: str) -> dict:
    """Return {'piano': [...], 'violin': [...], 'n_bars': int} for one MXL."""
    parts, soup = load_MusicXML(mxl_path)
    parts_info = inspect_score_parts(mxl_path)

    if len(parts) != len(parts_info):
        raise ValueError(
            f"part count mismatch: load_MusicXML={len(parts)} vs "
            f"score-part metadata={len(parts_info)}"
        )

    piano_idx  = next((i for i, p in enumerate(parts_info) if p['program'] == 0),  None)
    violin_idx = next((i for i, p in enumerate(parts_info) if p['program'] == 40), None)

    if piano_idx is None:
        raise ValueError("no piano part (midi-program 0)")
    if violin_idx is None:
        raise ValueError("no violin part (midi-program 40)")

    piano_tokens  = tokenize_single_part(parts[piano_idx],  soup, parts_info[piano_idx]['staves'])
    violin_tokens = tokenize_single_part(parts[violin_idx], soup, parts_info[violin_idx]['staves'])

    n_piano  = piano_tokens.count('bar')
    n_violin = violin_tokens.count('bar')
    if n_piano != n_violin:
        raise ValueError(f"bar count mismatch: piano={n_piano} violin={n_violin}")

    return {
        'piano':  piano_tokens,
        'violin': violin_tokens,
        'n_bars': n_piano,
    }


# ───────────────────────── batch driver ──────────────────────────────────────

def load_duet_csv_paths() -> dict[str, str]:
    """Map local mxl_duet/-relative path → original tracks string."""
    mapping = {}
    with open(CSV_PATH, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            tracks = (row.get('tracks') or '').strip()
            parts = tracks.split('-')
            if '0' not in parts or '40' not in parts:
                continue
            rel = (row.get('mxl') or '').strip().lstrip('./')
            if not rel.startswith('mxl/'):
                continue
            local_rel = rel[len('mxl/'):]
            mapping[local_rel] = tracks
    return mapping


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    paths_to_tracks = load_duet_csv_paths()
    print(f"Duet scores to tokenize: {len(paths_to_tracks):,}")
    print(f"Output dir: {OUT_DIR}\n")

    success = 0
    failed  = 0
    skipped = 0
    failure_reasons = Counter()

    for i, (local_rel, tracks) in enumerate(sorted(paths_to_tracks.items())):
        mxl_path = os.path.join(MXL_DIR, local_rel)
        if not os.path.exists(mxl_path):
            failed += 1
            failure_reasons['mxl missing'] += 1
            continue

        out_path = os.path.join(OUT_DIR, local_rel.replace('.mxl', '.json'))
        if os.path.exists(out_path):
            skipped += 1
            continue

        try:
            result = tokenize_duet_mxl(mxl_path)
        except AssertionError as e:
            failed += 1
            failure_reasons[f'assert: {e}'[:60]] += 1
            continue
        except Exception as e:
            failed += 1
            failure_reasons[f'{type(e).__name__}: {e}'[:60]] += 1
            continue

        result['tracks'] = tracks
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False)
        success += 1

        if (i + 1) % 100 == 0:
            print(f"  [{i+1:>5}/{len(paths_to_tracks)}]  ok={success:>5}  "
                  f"fail={failed:>4}  skip={skipped:>4}")

    print(f"\nDone.")
    print(f"  Tokenized successfully: {success:,}")
    print(f"  Already had output    : {skipped:,}")
    print(f"  Failed                : {failed:,}")
    print(f"  Total                 : {success + skipped + failed:,}")

    if failure_reasons:
        print(f"\n  Top failure reasons:")
        for reason, n in failure_reasons.most_common(10):
            print(f"    [{n:>4}] {reason}")


if __name__ == '__main__':
    main()
