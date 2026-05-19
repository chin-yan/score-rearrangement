"""
explore_duet_tokenize.py — Probe MXL structure for duet scores

The existing score_to_tokens.MusicXML_to_tokens() asserts len(parts) in (1, 2)
and labels the two parts as "R hand" / "L hand". That is wrong for piano+violin
duets (it would label violin as the left hand) and crashes outright for scores
with 3+ parts (e.g. 0-40-42 chamber arrangements).

This script:
  1. Picks a few sample duet MXLs from Phase06/mxl_duet/.
  2. Inspects the MusicXML <score-part> + <midi-instrument> elements to map
     each part to its MIDI program (0 = piano, 40 = violin, ...).
  3. Reports how many staves each part has (piano usually 2, violin 1).
  4. Tries tokenizing each part in isolation using a patched call to the
     existing helpers (measures_to_tokens / measure_to_tokens), to confirm we
     can produce per-track token streams.

The goal is to validate the path before writing tokenize_duet.py.
"""

import csv
import os
import sys
from collections import defaultdict

# Force UTF-8 console output (Windows default cp950 cannot encode
# part-names with Cyrillic / non-Latin characters).
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import zipfile
from bs4 import BeautifulSoup

# Import existing project helpers (they live one directory up)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from score_to_tokens import (
    load_MusicXML, measures_to_tokens, measure_to_tokens,
    common, others,
)

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
CSV_PATH    = os.path.join(PROJECT_DIR, "PDMX.csv")
MXL_DIR     = os.path.join(SCRIPT_DIR, "mxl_duet")


# ───────────────────────── sample selection ──────────────────────────────────

def find_sample_mxls(want=(("0-40", 2), ("0-40-40", 1), ("0-40-42", 1), ("0-40-41-42", 1))):
    """
    Return up to N paths per requested track-string pattern.

    `want` = tuple of (tracks_string, count_wanted).
    """
    found = defaultdict(list)
    target = {t: n for t, n in want}

    with open(CSV_PATH, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            tracks = (row.get('tracks') or '').strip()
            if tracks not in target:
                continue
            if len(found[tracks]) >= target[tracks]:
                continue
            rel = (row.get('mxl') or '').strip().lstrip('./')
            # CSV path: mxl/8/31/Qm....mxl  →  local: mxl_duet/8/31/Qm....mxl
            local = os.path.join(MXL_DIR, rel[len('mxl/'):]) if rel.startswith('mxl/') else None
            if local and os.path.exists(local):
                found[tracks].append(local)

    return found


# ───────────────────────── part / program inspection ─────────────────────────

def inspect_score_parts(mxl_path):
    """
    Parse just the <score-part> + <part-list> metadata, returning a list of
    dicts: {part_id, name, program (int or None), staves (int)}.
    """
    if mxl_path.lower().endswith('.mxl'):
        with zipfile.ZipFile(mxl_path, 'r') as z:
            xml_filename = next(
                (n for n in z.namelist() if n.endswith('.xml') and n != 'META-INF/container.xml'),
                None,
            )
            with z.open(xml_filename) as f:
                soup = BeautifulSoup(f, 'lxml-xml')
    else:
        with open(mxl_path, encoding='utf-8') as f:
            soup = BeautifulSoup(f, 'lxml-xml')

    # Map part id → program from <score-part>/<midi-instrument>
    part_meta = {}
    for sp in soup.find_all('score-part'):
        pid = sp.get('id')
        name_tag = sp.find('part-name')
        name = name_tag.text.strip() if name_tag else ''
        mi = sp.find('midi-instrument')
        prog = None
        if mi:
            mp = mi.find('midi-program')
            if mp:
                try:
                    prog = int(mp.text) - 1   # MusicXML midi-program is 1-indexed
                except ValueError:
                    pass
        part_meta[pid] = {'id': pid, 'name': name, 'program': prog}

    # Get staff counts from the actual <part> elements
    result = []
    for part in soup.find_all('part'):
        pid = part.get('id')
        meta = part_meta.get(pid, {'id': pid, 'name': '?', 'program': None}).copy()
        # number of distinct <staff> values in this part's notes
        staves = {s.text for s in part.find_all('staff')}
        meta['staves'] = len(staves) if staves else 1
        result.append(meta)

    return result


# ───────────────────────── per-part tokenization ─────────────────────────────

def tokenize_single_part(measures, soup, staves: int, note_name=True):
    """
    Tokenize ONE part in isolation.

    - If staves == 2 (piano): emit R/L hand split like the original tokenizer.
    - If staves == 1 (violin etc.): emit a single stream without R/L markers.

    Always bar-major. No chord-symbol layer (skipped for the duet pipeline).
    """
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


def explore(mxl_path):
    print(f"\n── {os.path.relpath(mxl_path, PROJECT_DIR)} ──")
    parts_info = inspect_score_parts(mxl_path)
    print(f"  {'id':<6} {'name':<24} {'program':>8} {'staves':>7}")
    for p in parts_info:
        print(f"  {p['id']:<6} {p['name'][:24]:<24} {str(p['program']):>8} {p['staves']:>7}")

    parts, soup = load_MusicXML(mxl_path)
    print(f"  load_MusicXML returned {len(parts)} parts; "
          f"existing MusicXML_to_tokens would {'work' if len(parts) in (1,2) else 'CRASH (assert)'}.")

    # Try per-part tokenization
    piano_idx  = next((i for i, p in enumerate(parts_info) if p['program'] == 0), None)
    violin_idx = next((i for i, p in enumerate(parts_info) if p['program'] == 40), None)
    if piano_idx is None or violin_idx is None:
        print("  Could not find both piano (prog=0) and violin (prog=40); skipping tokenization.")
        return

    piano_tokens = tokenize_single_part(parts[piano_idx], soup, parts_info[piano_idx]['staves'])
    violin_tokens = tokenize_single_part(parts[violin_idx], soup, parts_info[violin_idx]['staves'])

    print(f"  piano  part {piano_idx}: {len(piano_tokens):>5} tokens  | first 20: {piano_tokens[:20]}")
    print(f"  violin part {violin_idx}: {len(violin_tokens):>5} tokens  | first 20: {violin_tokens[:20]}")

    # quick sanity: bar count alignment
    pn = piano_tokens.count('bar')
    vn = violin_tokens.count('bar')
    print(f"  bars: piano={pn}  violin={vn}  ({'aligned' if pn == vn else 'MISMATCH'})")


def main():
    samples = find_sample_mxls()
    if not any(samples.values()):
        print(f"No samples found. Did extract_duet_mxl.py finish? Looking under {MXL_DIR}")
        return

    for tracks, paths in samples.items():
        print(f"\n========== tracks = {tracks!r} (showing {len(paths)} sample[s]) ==========")
        for p in paths:
            explore(p)


if __name__ == '__main__':
    main()
