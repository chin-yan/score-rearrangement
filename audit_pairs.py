"""
audit_pairs.py — Sample unique song pairs from pairs.jsonl and export to CSV.

Usage:
    python audit_pairs.py                          # default: 100 pairs → audit_pairs.csv
    python audit_pairs.py --n 200 --seed 0         # 200 pairs, different seed
    python audit_pairs.py --output my_audit.csv

Each row in the output represents one unique (src_path, tgt_path) song pair.
Columns are designed for a human reviewer to judge whether it is truly
the same song arranged at two different difficulty levels.
"""

import argparse
import csv
import json
import random
from collections import defaultdict


# ---------------------------------------------------------------------------
# Token parsing helpers (no external deps, works on raw token lists)
# ---------------------------------------------------------------------------

def split_into_bars(tokens):
    bars, cur = [], []
    for t in tokens:
        if t == 'bar':
            if cur:
                bars.append(cur)
            cur = []
        else:
            cur.append(t)
    if cur:
        bars.append(cur)
    return bars


def get_hand_tokens(bar_tokens, hand):
    try:
        idx = bar_tokens.index(hand)
    except ValueError:
        return []
    out, i = [], idx + 1
    while i < len(bar_tokens) and bar_tokens[i] not in ('R', 'L'):
        out.append(bar_tokens[i])
        i += 1
    return out


def first_n_notes(tokens, n_bars=2, hand='R'):
    """Return a readable string of note tokens from the first n_bars of one hand."""
    bars = split_into_bars(tokens)
    notes = []
    for bar in bars[:n_bars]:
        hand_toks = get_hand_tokens(bar, hand)
        notes += [t for t in hand_toks if t.startswith('note_') or t == 'rest']
    return ' '.join(notes) if notes else '(empty)'


def get_first_token(tokens, prefix):
    """Return the first token that starts with prefix, or ''."""
    for t in tokens:
        if t.startswith(prefix):
            return t
    return ''


def note_density(tokens):
    bars = split_into_bars(tokens)
    if not bars:
        return 0.0
    total = sum(1 for bar in bars for t in bar if t.startswith('note_'))
    return round(total / len(bars), 2)


def pitch_class_set(tokens, n_bars=4):
    """Jaccard-ready set of pitch step letters (C D E F G A B) from first n_bars."""
    bars = split_into_bars(tokens)
    pcs = set()
    for bar in bars[:n_bars]:
        for t in bar:
            if t.startswith('note_'):
                pcs.add(t[5])  # first letter after 'note_' is the step
    return pcs


def melodic_overlap(src_tokens, tgt_tokens, n_bars=4):
    """Jaccard similarity of pitch-class sets over first n_bars."""
    a = pitch_class_set(src_tokens, n_bars)
    b = pitch_class_set(tgt_tokens, n_bars)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return round(len(a & b) / len(a | b), 3)


def n_bars(tokens):
    return tokens.count('bar')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pairs',  default='data/pairs.jsonl')
    parser.add_argument('--output', default='audit_pairs.csv')
    parser.add_argument('--n',      type=int, default=100)
    parser.add_argument('--seed',   type=int, default=42)
    args = parser.parse_args()

    print(f"Loading {args.pairs} ...")
    with open(args.pairs, encoding='utf-8') as f:
        all_pairs = [json.loads(line) for line in f if line.strip()]
    print(f"  Total segments: {len(all_pairs):,}")

    # Deduplicate: keep one segment per unique (src_path, tgt_path) pair
    seen = {}
    for p in all_pairs:
        key = (p['src_path'], p['tgt_path'])
        if key not in seen:
            seen[key] = p
    unique_pairs = list(seen.values())
    print(f"  Unique song pairs: {len(unique_pairs):,}")

    random.seed(args.seed)
    sample = random.sample(unique_pairs, min(args.n, len(unique_pairs)))
    print(f"  Sampled: {len(sample)}")

    rows = []
    for i, p in enumerate(sample):
        src = p['src_tokens']
        tgt = p['tgt_tokens']

        src_key   = get_first_token(src, 'key_')
        tgt_key   = get_first_token(tgt, 'key_')
        src_time  = get_first_token(src, 'time_')
        tgt_time  = get_first_token(tgt, 'time_')
        key_match  = 'YES' if src_key == tgt_key else 'NO'
        time_match = 'YES' if src_time == tgt_time else 'NO'

        overlap = melodic_overlap(src, tgt, n_bars=4)

        # flag suspicious pairs automatically for easy filtering
        suspicious = (
            key_match  == 'NO' or
            time_match == 'NO' or
            overlap    < 0.4
        )

        rows.append({
            'id':              i + 1,
            'song':            p['song'],
            'src_level':       p['src_level'],
            'tgt_level':       p['tgt_level'],
            'src_key':         src_key,
            'tgt_key':         tgt_key,
            'key_match':       key_match,
            'src_time':        src_time,
            'tgt_time':        tgt_time,
            'time_match':      time_match,
            'src_density':     note_density(src),
            'tgt_density':     note_density(tgt),
            'src_bars':        n_bars(src),
            'tgt_bars':        n_bars(tgt),
            'melodic_overlap': overlap,   # 0.0~1.0; <0.4 likely different songs
            'suspicious':      'YES' if suspicious else 'no',
            'same_song_judgement': '',    # ← fill in manually: YES / NO / UNSURE
            'notes':           '',        # ← free text
            'src_first_notes': first_n_notes(src, n_bars=2, hand='R'),
            'tgt_first_notes': first_n_notes(tgt, n_bars=2, hand='R'),
            'src_path':        p['src_path'],
            'tgt_path':        p['tgt_path'],
        })

    fieldnames = [
        'id', 'song', 'src_level', 'tgt_level',
        'src_key', 'tgt_key', 'key_match',
        'src_time', 'tgt_time', 'time_match',
        'src_density', 'tgt_density',
        'src_bars', 'tgt_bars',
        'melodic_overlap',
        'suspicious',
        'same_song_judgement',  # ← you fill this in
        'notes',
        'src_first_notes', 'tgt_first_notes',
        'src_path', 'tgt_path',
    ]

    with open(args.output, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    n_suspicious = sum(1 for r in rows if r['suspicious'] == 'YES')
    print(f"\nDone → {args.output}")
    print(f"  Auto-flagged as suspicious: {n_suspicious} / {len(rows)} "
          f"({100*n_suspicious/len(rows):.1f}%)")
    print(f"\nColumn guide:")
    print(f"  melodic_overlap  : 1.0 = identical pitch classes, 0.0 = nothing in common")
    print(f"  suspicious       : YES if key/time mismatch OR overlap < 0.4")
    print(f"  same_song_judgement : fill in yourself after reviewing src/tgt_first_notes")


if __name__ == '__main__':
    main()
