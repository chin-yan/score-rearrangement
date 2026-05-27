"""
analyze_pairs.py — Full statistical analysis of all unique song pairs in pairs.jsonl.

Runs over every unique (src_path, tgt_path) pair (not per-segment) and produces:
  1. Console summary (contamination rate, breakdown by problem type)
  2. analyze_pairs_report.csv — one row per unique pair, with quality flags

Usage:
    python analyze_pairs.py                        # reads data/pairs.jsonl
    python analyze_pairs.py --pairs data/pairs.jsonl --output report.csv
"""

import argparse
import csv
import json
from collections import defaultdict


# ---------------------------------------------------------------------------
# Known bad song name patterns
# ---------------------------------------------------------------------------

# Song names that are definitely garbage labels
BLACKLIST_EXACT = {
    'misc', 'untitled', 'new score', 'test', 'unknown', 'na', '',
    'new composition', 'untitled score', 'my score',
}

# Substrings that indicate a template / tool artifact (case-insensitive)
BLACKLIST_SUBSTR = [
    'abcm2ps', 'sample tune', 'sample3', 'staff break',
    'musescore', 'finale', 'sibelius', 'noteworthy',
    'template', 'default', 'exercise', 'etude test',
]

# Collection / opus titles — these are BOOKS, not individual pieces.
# Different users upload different pieces from the same book under the same name.
COLLECTION_SUBSTR = [
    'sz.', 'op.', 'bwv ', 'k.', 'woo ', 'hob.', 'd.', 'l.',  # catalogue numbers
    'book ', 'vol.', 'volume ', 'collection', 'album',
    'for children', 'notebook', 'etudes', 'études',
    'preludes', 'variations on', 'sonatas',
]


def song_name_flags(name: str) -> dict:
    """Return a dict of boolean quality flags for a song name."""
    n = name.strip().lower()
    return {
        'name_blacklisted': (
            n in BLACKLIST_EXACT or
            any(s in n for s in BLACKLIST_SUBSTR)
        ),
        'name_is_collection': any(s in n for s in COLLECTION_SUBSTR),
        'name_too_short':     len(n) <= 3,
        'name_too_generic':   n in {
            'minuet', 'waltz', 'march', 'song', 'melody', 'tune',
            'piece', 'music', 'piano', 'score', 'hymn', 'aria',
            'dance', 'carol', 'polka', 'mazurka', 'nocturne',
        },
    }


# ---------------------------------------------------------------------------
# Token helpers
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


def get_first_token(tokens, prefix):
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


def pitch_letters(tokens, n_bars=4):
    """Set of pitch step letters (C D E F G A B) from first n_bars."""
    bars = split_into_bars(tokens)
    return {t[5] for bar in bars[:n_bars] for t in bar if t.startswith('note_')}


def pitch_ngrams(tokens, n=3, n_bars=4):
    """Set of pitch-letter n-grams from the first n_bars (order-sensitive)."""
    bars = split_into_bars(tokens)
    notes = []
    for bar in bars[:n_bars]:
        for hand in ('R', 'L'):
            notes += [t[5] for t in get_hand_tokens(bar, hand) if t.startswith('note_')]
        if not notes:
            notes += [t[5] for t in bar if t.startswith('note_')]
    if len(notes) < n:
        return set()
    return set(zip(*[notes[i:] for i in range(n)]))


def melodic_letter_overlap(src, tgt, n_bars=4):
    """Jaccard on pitch-letter sets (same as audit_pairs.py)."""
    a, b = pitch_letters(src, n_bars), pitch_letters(tgt, n_bars)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return round(len(a & b) / len(a | b), 3)


def melodic_ngram_overlap(src, tgt, n=3, n_bars=4):
    """Jaccard on pitch trigram sets (order-sensitive, harder to fool)."""
    a, b = pitch_ngrams(src, n, n_bars), pitch_ngrams(tgt, n, n_bars)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return round(len(a & b) / len(a | b), 3)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pairs',  default='data/pairs.jsonl')
    parser.add_argument('--output', default='analyze_pairs_report.csv')
    args = parser.parse_args()

    print(f"Loading {args.pairs} ...")
    with open(args.pairs, encoding='utf-8') as f:
        all_segs = [json.loads(line) for line in f if line.strip()]
    print(f"  Total segments: {len(all_segs):,}")

    # Deduplicate: one row per unique (src_path, tgt_path)
    seen = {}
    for p in all_segs:
        key = (p['src_path'], p['tgt_path'])
        if key not in seen:
            seen[key] = p
    unique_pairs = list(seen.values())
    print(f"  Unique directional pairs: {len(unique_pairs):,}")
    print(f"  Processing all pairs — this may take a few minutes...")

    rows = []
    problem_counts = defaultdict(int)
    overlap_letter_dist = defaultdict(int)   # bucket by 0.1
    overlap_ngram_dist  = defaultdict(int)

    for i, p in enumerate(unique_pairs):
        if i % 5000 == 0:
            print(f"    [{i:,}/{len(unique_pairs):,}]")

        src, tgt = p['src_tokens'], p['tgt_tokens']
        song = p['song']

        src_key  = get_first_token(src, 'key_')
        tgt_key  = get_first_token(tgt, 'key_')
        src_time = get_first_token(src, 'time_')
        tgt_time = get_first_token(tgt, 'time_')

        key_mismatch  = (src_key != tgt_key and src_key and tgt_key)
        time_mismatch = (src_time != tgt_time and src_time and tgt_time)

        ol = melodic_letter_overlap(src, tgt)
        on = melodic_ngram_overlap(src, tgt)

        name_flags = song_name_flags(song)

        # Determine problem type (hierarchical — pick the most severe)
        if name_flags['name_blacklisted']:
            problem = 'blacklisted_name'
        elif name_flags['name_is_collection']:
            problem = 'collection_title'
        elif key_mismatch or time_mismatch:
            problem = 'key_or_time_mismatch'
        elif name_flags['name_too_generic']:
            problem = 'generic_name'
        elif on < 0.2:
            problem = 'low_ngram_overlap'
        elif ol < 0.4:
            problem = 'low_letter_overlap'
        else:
            problem = 'ok'

        is_problematic = (problem != 'ok')
        if is_problematic:
            problem_counts[problem] += 1

        # Distribution buckets
        bucket_l = round(ol * 10) / 10   # e.g. 0.857 → 0.9
        bucket_n = round(on * 10) / 10
        overlap_letter_dist[bucket_l] += 1
        overlap_ngram_dist[bucket_n]  += 1

        rows.append({
            'src_path':          p['src_path'],
            'tgt_path':          p['tgt_path'],
            'song':              song,
            'src_level':         p['src_level'],
            'tgt_level':         p['tgt_level'],
            'src_key':           src_key,
            'tgt_key':           tgt_key,
            'key_mismatch':      'YES' if key_mismatch else 'no',
            'src_time':          src_time,
            'tgt_time':          tgt_time,
            'time_mismatch':     'YES' if time_mismatch else 'no',
            'src_density':       note_density(src),
            'tgt_density':       note_density(tgt),
            'melodic_overlap_letter': ol,
            'melodic_overlap_ngram':  on,
            'name_blacklisted':  'YES' if name_flags['name_blacklisted'] else 'no',
            'name_is_collection':'YES' if name_flags['name_is_collection'] else 'no',
            'name_too_generic':  'YES' if name_flags['name_too_generic'] else 'no',
            'problem_type':      problem,
            'is_problematic':    'YES' if is_problematic else 'no',
        })

    # Write CSV
    fieldnames = [
        'song', 'src_level', 'tgt_level',
        'src_key', 'tgt_key', 'key_mismatch',
        'src_time', 'tgt_time', 'time_mismatch',
        'src_density', 'tgt_density',
        'melodic_overlap_letter', 'melodic_overlap_ngram',
        'name_blacklisted', 'name_is_collection', 'name_too_generic',
        'problem_type', 'is_problematic',
        'src_path', 'tgt_path',
    ]
    with open(args.output, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # -----------------------------------------------------------------------
    # Console summary
    # -----------------------------------------------------------------------
    total = len(unique_pairs)
    total_segs = len(all_segs)
    n_ok   = sum(1 for r in rows if r['problem_type'] == 'ok')
    n_bad  = total - n_ok

    print(f"\n{'='*60}")
    print(f"ANALYSIS SUMMARY")
    print(f"{'='*60}")
    print(f"Total segments in pairs.jsonl : {total_segs:>10,}")
    print(f"Unique directional pairs      : {total:>10,}")
    print(f"  Clean (ok)                  : {n_ok:>10,}  ({100*n_ok/total:.1f}%)")
    print(f"  Problematic                 : {n_bad:>10,}  ({100*n_bad/total:.1f}%)")
    print(f"\nBreakdown by problem type:")
    for ptype, count in sorted(problem_counts.items(), key=lambda x: -x[1]):
        print(f"  {ptype:<30} {count:>8,}  ({100*count/total:.1f}%)")

    print(f"\nMelodic overlap (letter-Jaccard) distribution:")
    for b in sorted(overlap_letter_dist):
        bar_w = int(50 * overlap_letter_dist[b] / total)
        print(f"  {b:.1f}  {'█'*bar_w:<50}  {overlap_letter_dist[b]:,}")

    print(f"\nMelodic overlap (trigram-Jaccard) distribution:")
    for b in sorted(overlap_ngram_dist):
        bar_w = int(50 * overlap_ngram_dist[b] / total)
        print(f"  {b:.1f}  {'█'*bar_w:<50}  {overlap_ngram_dist[b]:,}")

    print(f"\nTop 20 most common song names (potential collection/generic issues):")
    song_counts = defaultdict(int)
    for r in rows:
        song_counts[r['song']] += 1
    for name, cnt in sorted(song_counts.items(), key=lambda x: -x[1])[:20]:
        flags = []
        nf = song_name_flags(name)
        if nf['name_blacklisted']:   flags.append('BLACKLIST')
        if nf['name_is_collection']: flags.append('COLLECTION')
        if nf['name_too_generic']:   flags.append('GENERIC')
        flag_str = f"  ← {', '.join(flags)}" if flags else ''
        print(f"  {cnt:>6,}x  {name}{flag_str}")

    print(f"\nOutput → {args.output}")
    print(f"\nConclusion:")
    if n_bad / total > 0.15:
        print(f"  ⚠  {100*n_bad/total:.1f}% problematic pairs — STRONGLY recommend regenerating pairs.jsonl")
    elif n_bad / total > 0.05:
        print(f"  ⚠  {100*n_bad/total:.1f}% problematic pairs — recommend regenerating pairs.jsonl")
    else:
        print(f"  ✓  {100*n_bad/total:.1f}% problematic — dataset is relatively clean")


if __name__ == '__main__':
    main()
