"""
validate_duet_data.py — Verify Phase 6 data assumptions against PDMX.csv

Confirms:
  1. Count of duet scores (tracks contain both '0' piano AND '40' violin).
  2. Distribution of track combinations among those duet scores.
  3. Song-length distribution and rough segment-count estimate for
     reverse-augmented training data.
"""

import os
import csv
from collections import Counter, defaultdict

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)        # one level up: project root
CSV_PATH    = os.path.join(PROJECT_DIR, "PDMX.csv")

SEG_MIN    = 4   # min bars per segment (matches build_pairs.py)
SEG_MAX    = 8   # max bars per segment
SEG_STRIDE = 2   # sliding-window stride


def has_piano_and_violin(tracks_str: str) -> bool:
    """Return True if the tracks list contains BOTH program 0 and program 40."""
    parts = tracks_str.split('-')
    return '0' in parts and '40' in parts


def estimate_segments(n_bars: int) -> int:
    """Sliding-window segment count (mirrors generate_segments in build_pairs.py)."""
    if n_bars < SEG_MIN:
        return 0
    n_steps = max(0, (n_bars - SEG_MIN) // SEG_STRIDE + 1)
    return n_steps


def main():
    if not os.path.exists(CSV_PATH):
        print(f"ERROR: {CSV_PATH} not found.")
        return

    total_rows         = 0
    duet_rows          = 0
    track_combo_counts = Counter()
    bar_counts         = []
    duet_song_names    = defaultdict(int)

    other_program_counts  = Counter()   # which extra programs show up alongside 0+40
    extra_track_breakdown = Counter()   # 'exactly_0-40', 'multi_violin', 'has_other_inst'

    with open(CSV_PATH, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_rows += 1
            tracks_str = (row.get('tracks') or '').strip()
            if not tracks_str:
                continue

            if not has_piano_and_violin(tracks_str):
                continue

            duet_rows += 1
            track_combo_counts[tracks_str] += 1

            parts = tracks_str.split('-')
            programs = Counter(parts)
            other = {p for p in programs if p not in ('0', '40')}
            if not other:
                if programs['0'] == 1 and programs['40'] == 1:
                    extra_track_breakdown['exactly one piano + one violin (e.g. 0-40)'] += 1
                else:
                    extra_track_breakdown['piano+violin doublings only (e.g. 0-40-40, 0-0-40)'] += 1
            else:
                extra_track_breakdown['contains other instruments (e.g. 0-40-41-42)'] += 1
                for p in other:
                    other_program_counts[p] += 1

            try:
                n_bars = int(float(row.get('song_length.bars', '0') or 0))
            except ValueError:
                n_bars = 0
            if n_bars > 0:
                bar_counts.append(n_bars)

            song = (row.get('song_name') or '').strip()
            if song and song != 'NA':
                duet_song_names[song] += 1

    # ── Report ────────────────────────────────────────────────────────────────
    print(f"PDMX.csv total rows:                          {total_rows:>8,}")
    print(f"Duet scores (tracks contain BOTH 0 and 40):   {duet_rows:>8,}")
    print(f"  (spec says 1,629 — {'MATCH' if duet_rows == 1629 else 'MISMATCH'})")
    print()

    print("── Track-combination breakdown ──────────────────────────────────")
    for label, count in extra_track_breakdown.most_common():
        pct = 100 * count / duet_rows if duet_rows else 0
        print(f"  {label:<55} {count:>6,}  ({pct:5.1f}%)")
    print()

    print("── Top 15 exact `tracks` strings ────────────────────────────────")
    for combo, count in track_combo_counts.most_common(15):
        pct = 100 * count / duet_rows if duet_rows else 0
        print(f"  {combo:<30} {count:>6,}  ({pct:5.1f}%)")
    print()

    if other_program_counts:
        print("── Other MIDI programs co-occurring with piano+violin ───────────")
        for prog, count in other_program_counts.most_common(15):
            print(f"  program {prog:>3}                   appears in {count:>5,} duet scores")
        print()

    print("── Unique songs (by song_name) among duet scores ────────────────")
    print(f"  Unique song titles:                          {len(duet_song_names):>6,}")
    multi_arr = sum(1 for v in duet_song_names.values() if v >= 2)
    print(f"  Songs with 2+ duet arrangements:             {multi_arr:>6,}")
    print()

    if bar_counts:
        bar_counts.sort()
        total = len(bar_counts)
        mean  = sum(bar_counts) / total
        med   = bar_counts[total // 2]
        p10   = bar_counts[int(total * 0.10)]
        p90   = bar_counts[int(total * 0.90)]
        usable = sum(1 for b in bar_counts if b >= SEG_MIN)
        total_segments = sum(estimate_segments(b) for b in bar_counts)

        print("── Song-length distribution (bars) ──────────────────────────────")
        print(f"  Mean / median / p10 / p90:   "
              f"{mean:6.1f} / {med:>5} / {p10:>5} / {p90:>5}")
        print(f"  Scores with >= {SEG_MIN} bars (segmentable):  {usable:>6,}  "
              f"({100*usable/total:.1f}% of duets)")
        print()
        print("── Reverse-augmentation training-data estimate ──────────────────")
        print(f"  Segmenting params: SEG_MIN={SEG_MIN}, SEG_STRIDE={SEG_STRIDE}")
        print(f"  Estimated segments per dataset (A or B):  ~{total_segments:>8,}")
        print(f"  Dataset A + B combined:                   ~{2*total_segments:>8,}")
        print()
        print(f"  Reference: Phase 1.2 piano-only produced ~600k+ segments.")
        print(f"  Paper trained on 130,930 segment pairs.")


if __name__ == '__main__':
    main()
