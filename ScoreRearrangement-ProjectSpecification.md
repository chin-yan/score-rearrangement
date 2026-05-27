# Piano Score Rearrangement — Project Specification

---

## 1. Project Overview

This project implements an end-to-end piano score rearrangement system that transforms a piano score into a target difficulty level (Beginner / Elementary / Intermediate / Advanced).

The approach is based on the paper:
> "Piano Score Rearrangement into Multiple Difficulty Levels via Notation-to-Notation Approach"
> Masahiro Suzuki, EURASIP Journal on Audio, Speech, and Music Processing, 2023.

The system operates entirely at the notation level (musical symbols, articulations, beams, ties) rather than the note/MIDI level, preserving musical expressiveness. It uses the ST+ token representation and a sequence-to-sequence (seq2seq) Transformer model conditioned on difficulty level tokens.

---

## 2. Dataset

- **Source:** PDMX (Piano Data from MuseScore eXchange)
- **Total piano scores:** ~205,789 (filtered from 254,035 total by MIDI program = 0)
- **Songs with 2+ arrangements:** ~31,875 (used as training pairs)

Difficulty labels are **not pre-tagged** in PDMX and must be computed from token features:

| Level | Name | Definition |
|---|---|---|
| Level 1 | Beginner | Max 1 simultaneous note per hand |
| Level 2 | Elementary | Max 2 simultaneous notes per hand |
| Level 3 | Intermediate | Max 3 simultaneous notes per hand |
| Level 4 | Advanced | No restriction |

Supporting metrics (also computed from tokens):
- **Note density:** number of notes per measure
- **Pitch width:** semitone range (highest – lowest pitch) per measure
- **Polyphony:** max simultaneous notes per measure

---

### Data Quality Investigation (Round 1 → Round 2)

#### Problem observed after Round 1 training

The Round 1 model (60 epochs, val_loss = 1.24) successfully changed the difficulty level of the output, but the output music did not sound like the same song as the input — it sounded like a completely different piece. This indicated the model had learned **style transfer** ("what Lv.1 music sounds like") rather than **content-preserving transformation** ("how to simplify this specific melody").

#### Root cause analysis

PDMX `song_name` is a user-entered free-text field. Two scores sharing the same `song_name` do not necessarily share the same melody. The original compatibility filter (key + time + density) was not sufficient because:

1. **Key/time/density can match by coincidence** — two unrelated pieces in C major with 4/4 time and similar note count both pass all three checks.
2. **The letter-based Jaccard overlap metric has a blind spot** — in C major, any two pieces share the same pitch-class letter set {C, D, E, F, G, A, B}, giving an overlap of 1.0 even if the melodies are completely different.
3. **Collection titles are mislabeled as single songs** — "For Children Sz.42" is a Bartók book of 85 individual pieces; different users uploaded different pieces from it under the same title, creating false pairs.
4. **Garbage song names** — labels like "misc", "Sample tune - abcm2ps", "untitled" match completely unrelated scores.

#### Diagnostic process

Two analysis tools were written to quantify the contamination:

**`audit_pairs.py`** — samples N unique song pairs from `pairs.jsonl` and exports a human-readable CSV for manual review. Each row shows the song name, difficulty levels, key/time, note density, letter-Jaccard overlap, trigram-Jaccard overlap, and the first two bars of each hand for direct comparison. A `suspicious` flag is auto-set when key/time mismatches or overlap < 0.4.

**`analyze_pairs.py`** — runs over all unique pairs in `pairs.jsonl` and produces a full statistical report: contamination rate, breakdown by problem type, overlap distributions, and top song names. Problem types detected:
- `low_ngram_overlap` — trigram melodic similarity < 0.2 (order-sensitive, catches false positives that letter-Jaccard misses)
- `collection_title` — song name contains known collection identifiers (Sz.42, Op.28, etc.)
- `blacklisted_name` — garbage labels (misc, sample tune, untitled, etc.)
- `generic_name` — overly common titles (Minuet, Waltz, etc.) with no distinguishing information

#### Round 1 contamination findings

Running `analyze_pairs.py` on the Round 1 `pairs.jsonl`:

| Metric | Value |
|---|---|
| Total segments | 25,600 |
| Unique directional pairs | 2,468 |
| **Contamination rate** | **34.9%** |
| `low_ngram_overlap` | 620 pairs (25.1%) |
| `collection_title` | 174 pairs (7.1%) |
| `blacklisted_name` | 36 pairs (1.5%) |
| `generic_name` | 32 pairs (1.3%) |

The trigram overlap distribution showed that 58% of all pairs had overlap ≤ 0.3, confirming that the majority of pairs did not share the same melody.

#### Fixes applied to `build_pairs.py`

Three changes were made to produce cleaner pairs:

**1. Melodic trigram overlap filter (`NGRAM_OVERLAP_MIN = 0.25`)**

Added `melodic_ngram_overlap()` which computes the Jaccard similarity of pitch-letter trigram sets from the first 4 bars of each arrangement's right-hand melody. Unlike the letter-set Jaccard, trigrams are order-sensitive — `(C,E,G)` and `(G,E,C)` are different trigrams — so two different songs in the same key can no longer both score 1.0. Pairs where this score is below 0.25 are rejected.

**2. Song name blacklist and collection filter (`is_bad_song_name()`)**

Added a function that rejects scores whose `song_name` is a known garbage label (`misc`, `untitled`, `sample tune`, `abcm2ps`-generated files, etc.) or a known collection title (`For Children Sz.42`, `Mikrokosmos`) where the same title covers many different individual pieces.

**3. Relaxed bar count tolerance (`BAR_TOLERANCE 0.1 → 0.15`)**

The previous 10% tolerance was slightly too strict and rejected some legitimate pairs where one arrangement had a pickup bar or a brief intro absent in the other, causing a small bar count difference. Relaxing to 15% recovers these pairs without significantly increasing false positives.

#### Round 2 pair quality results

After regenerating `pairs.jsonl` with the updated `build_pairs.py`:

| Metric | Round 1 | Round 2 |
|---|---|---|
| Total segments | 25,600 | **19,172** |
| Unique directional pairs | 2,468 | **1,854** |
| Contamination rate | 34.9% | **10.9%** |
| Clean pairs | 1,606 (65.1%) | **1,652 (89.1%)** |
| `low_ngram_overlap` | 620 (25.1%) | 106 (5.7%) |
| `blacklisted_name` | 36 (1.5%) | **0 (0%)** |

The remaining 10.9% flagged pairs are mostly false positives from the collection detector over-triggering on single works that happen to have opus numbers (e.g. "Symphony No.9 Op.125" is a single work, not a collection). The true contamination rate is estimated at **< 5%**.

Fewer total segments (19,172 vs 25,600) is expected and acceptable — the removed pairs were noise. 19,172 high-quality segments is sufficient for training and considerably better than 25,600 segments with 35% contamination.

---

## 2.1 Cross-Instrument Extension: Data Analysis

We investigated the feasibility of extending this system to cross-instrument translation (piano ↔ violin) with combined difficulty transformation. The following data was found in PDMX:

| | Count |
|---|---|
| Songs with 2+ piano arrangements | 31,875 |
| Songs with purely violin-only arrangements | 434 |
| Songs with BOTH piano AND violin | 141 |
| Total piano+violin cross pairs | 1,381 |

141 songs is far too little to train a cross-instrument model. For reference, the paper trained on 1,957 scores and got 130,930 segment pairs. With only 1,381 cross pairs, you would get approximately 10,000–20,000 segments after chunking — likely not enough for the model to generalize.

**Decision: implement piano-only first, for the following reasons:**

1. **Data is solid** — 31,875 multi-arrangement songs gives plenty of training pairs.
2. **Validates the pipeline** — confirms the full stack works before tackling a harder problem.
3. **Almost no rework** — when adding cross-instrument later, only `tokenize_all.py`, `build_pairs.py`, and conditioning tokens in `model.py` need changes. The rest stays the same.
4. **Cross-instrument needs more data** — additional violin+piano paired data from other sources (e.g., IMSLP) would be needed before expanding.

The cross-instrument extension is planned as **Phase 6** (see Section 5).

---

## 3. System Architecture

The full pipeline:

```
[Input MXL]
   » score_to_tokens.py   ——  tokenize to ST+ format
   » prepend {Dsrc, Dtgt}  ——  difficulty conditioning tokens
   » seq2seq model          ——  encoder-decoder Transformer
   » strip conditioning     ——  remove Dtgt prefix from output
   » tokens_to_score.py   ——  detokenize to music21 Score
[Output MXL]
```

**Difficulty Conditioning (from paper Fig. 2b):**
- Source sequence: `Dsrc Dtgt bar key_flat_1 time_4/4 R ...`
- Target sequence: `Dtgt bar key_flat_1 time_4/4 R ...`

Score pairs are trained **bidirectionally** (easier→harder and harder→easier), and all nC2 combinations of available arrangements per song are used as training pairs.

---

## 4. Model

| Property | Value |
|---|---|
| Architecture | Encoder-Decoder Transformer |
| Model size | ~0.3M parameters (small, matching paper config) |
| Embedding dim | 48 |
| FFN dim | 96 |
| Layers | 3 encoder + 3 decoder |
| Seq length | 4–8 measure segments (overlapping) |
| Augmentation | Pitch transposition ±2 semitones (training only) |

**Vocabulary:** ST+ tokens (`bar`, `R`, `L`, `clef_*`, `key_*`, `time_*`, `note_*`, `len_*`, `stem_*`, `beam_*`, `tie_*`, `rest`, `accent`, `staccato`, `tenuto`, `slur_start`, `slur_stop`, `chord_*`, `bass_*`, `<voice>`, `</voice>`) + special tokens (`<pad>`, `<sos>`, `<eos>`, `Lv.1`, `Lv.2`, `Lv.3`, `Lv.4`)

---

## 5. Project Breakdown

### Phase 1 — Data Preparation

**[1.1] `tokenize_all.py`**
- Filter PDMX to piano-only scores (program=0) via PDMX.csv
- Run `MusicXML_to_tokens()` on all MXL files
- Save token sequences as JSON under `tokens/`
- **Status: DONE** (199,021 / ~205,789 scores tokenized, 96.7% coverage)

**[1.2] `build_pairs.py`**
- Compute difficulty metrics (polyphony, note density, pitch width) from token files
- Assign Lv.1–Lv.4 labels per score
- Match same-song scores using `song_name` column in PDMX.csv
- Generate all nC2 bidirectional pairs
- Segment pairs into 4–8-bar chunks with overlap:
  - **Breaks long songs into short chunks** — the model only processes 4–8 bars at a time, keeping sequences short enough for the small (0.3M) model to handle efficiently.
  - **Overlapping windows multiply training data** — one song pair generates ~28 segments instead of 1, helping the model generalize despite having limited songs.

  ```
  Input song (60 bars)
      ↓ split into segments
  [bar1-6]   → model → [bar1-6 at Lv.1]
  [bar7-12]  → model → [bar7-12 at Lv.1]
  [bar13-18] → model → [bar13-18 at Lv.1]
      ↓ stitch back together
  Output song (60 bars, Lv.1)
  ```

- Compatibility filters (all must pass):
  - Same key signature — different keys almost certainly means different arrangements
  - Same time signature — structurally incompatible otherwise
  - Note density ratio ≤ 3× — if one arrangement has far more notes/bar, likely unrelated
  - **Melodic trigram overlap ≥ 0.25** *(added Round 2)* — order-sensitive pitch n-gram similarity; catches songs that share key/time/density but have different melodies
  - **Song name blacklist** *(added Round 2)* — rejects garbage labels and known collection titles where the same name covers many distinct pieces
  - **BAR_TOLERANCE = 0.15** *(relaxed from 0.1 in Round 2)* — recovers pairs with pickup bars or small structural differences
- Save as `pairs.jsonl`
- **Status: DONE** (Round 2 pairs: 19,172 segments, ~89% clean)

**[1.3] `build_vocab.py`**
- Scan all token files to collect unique tokens
- Add special tokens: `<pad>`, `<sos>`, `<eos>`, `Lv.1`, `Lv.2`, `Lv.3`, `Lv.4`
- Save `vocab.json` (token → index mapping)
- **Status: DONE**

---

### Phase 2 — Model Implementation

**[2.1] `model.py`**
- Encoder-Decoder Transformer (PyTorch)
- Shared token embedding for source and target
- Difficulty conditioning via prepended `Lv.*` tokens
- Key design decisions:
    - Shared embedding — single nn.Embedding for both encoder and decoder inputs
    - Difficulty conditioning — handled entirely by prepended Lv.* tokens in the sequence; no special architecture needed inside the model
    - `_bool_to_additive` — converts bool padding masks to float additive masks (-inf) so they're consistent with the float causal mask PyTorch generates, avoiding deprecation warnings
    - `forward()` — teacher-forced training path (src + tgt-shifted-right → logits)
    - `encode()` / `decode_step()` — separated for autoregressive inference
    - `greedy_decode()` — batched decoding used by `infer.py`:
        - `init_token_idx` — forces Dtgt as the first decoder output token (prevents the model from predicting the wrong difficulty level)
        - `temperature` — softmax temperature for sampling (>1 adds variety, <1 sharpens)
        - `top_k` — if >0, samples from top-k logits instead of argmax, breaking repetitive collapse
- **Status: DONE**

**[2.2] `dataset_seq2seq.py`**
- PyTorch Dataset that loads `pairs.jsonl`
- Encodes tokens using `vocab.json`
- Applies pitch augmentation (±2 semitones)
- Pads and batches source/target sequences
- `transpose_tokens(tokens, shift)`
    - Transposes all pitch-bearing tokens by shift semitones:
        - `note_*` — MIDI number ± shift, back to letter name
        - `key_*` — circle-of-fifths shift (e.g. G major +1 → Ab major)
        - `bass_*` / `chord_*` — pitch-class rotation, quality unchanged
- `ScorePairDataset`
    - Loads all pairs from pairs.jsonl
    - Builds encoder/decoder sequences per paper Fig. 2b: `src = [Dsrc, Dtgt, …, <eos>]`, `tgt = [<sos>, Dtgt, …, <eos>]`
    - On each `__getitem__` randomly samples a shift from {-2,-1,0,1,2}
- `make_collate_fn(pad_id)`
    - Returns a collate function that pads and splits the target into:
        - `tgt_in = tgt[:-1]` (decoder input, teacher-forced)
        - `tgt_out = tgt[1:]` (cross-entropy target)
- `make_splits(pairs_path, vocab_path)`
    - Song-level train/val split so no song leaks across splits (default 5% val)
- **Status: DONE**

---

### Phase 3 — Training

**[3.1] `train_seq2seq.py`**
- Training loop with Adam optimizer + LR warmup/decay
- Teacher forcing on target sequence
- Validation loss tracking, early stopping
- Checkpoint saving (best model by validation loss)
- LR schedule — `make_lr_lambda`: linear warmup over `--warmup_steps` steps, then cosine decay to `--min_lr`
- Training loop
    - Teacher forcing: `tgt_in = tgt[:-1]` → model → compared against `tgt_out = tgt[1:]`
    - `F.cross_entropy` with `ignore_index=pad_id` (pad positions don't contribute to loss)
    - `label_smoothing=0.1` (helps regularization on a small dataset)
    - Gradient clipping at `grad_clip=1.0`
    - Gradient accumulation (`--accum_steps`, default 4) — effective batch = batch_size × accum_steps without extra VRAM
    - Per-batch tqdm bar showing running loss + current LR
- Checkpointing
    - `best.pt` — saved whenever val loss improves (stores model, optimizer, scheduler state for resuming)
    - `epoch_NNNN.pt` — periodic snapshot every `--save_every` epochs (default 10)
    - `train_log.csv` — append-only CSV with epoch, train loss, val loss, lr, elapsed time
- Early stopping — stops after `--patience` consecutive epochs (default 10) with no val improvement
- Resuming — `--resume data/checkpoints/best.pt` restores full state and continues from next epoch

**Round 1 results (noisy pairs — 34.9% contamination):**
- Trained 60 epochs, val_loss = 1.24, LR decayed to minimum (1e-5)
- Output changed difficulty level correctly but sounded like a different song entirely
- Root cause confirmed by pair quality analysis: model learned style distribution of each level, not melody-preserving transformation, because most training pairs did not share the same melody

**Round 2 (clean pairs — ~5% contamination):**
- Regenerated `pairs.jsonl` using updated `build_pairs.py` (trigram filter + blacklist + relaxed BAR_TOLERANCE)
- Training from scratch: `python train_seq2seq.py --epochs 100 --lr 1e-3`
- Training from scratch (not resuming) required since data distribution changed significantly
- **Status: IN PROGRESS**

---

### Phase 4 — Inference

**[4.1] `infer.py`**
- Load trained model checkpoint
- Accept input MXL + target difficulty level
- Tokenize – prepend conditioning – run model – detokenize
- Output rearranged MXL file

- Pipeline:
  ```
  Input MXL
     ↓ MusicXML_to_tokens()     tokenize to ST+ format
     ↓ split_into_bars()        split into per-bar lists
     ↓ assign_level()           detect source difficulty (Lv.1–4)
     ↓ non-overlapping chunks   --seg_len bars each (default 8)
     ↓ encode_segment()         prepend [Dsrc, Dtgt, ..., <eos>]
     ↓ model.greedy_decode()    autoregressive generation
     ↓ strip Dtgt token         remove conditioning prefix from output
     ↓ concatenate segments     stitch all bars back together
     ↓ tokens_to_score()        music21 Score
     ↓ score.write('musicxml')  output .mxl
  Output MXL
  ```

- Key arguments:
  ```
  --input       Input MXL or XML file (required)
  --output      Output MXL file (required)
  --level       Target difficulty: Lv.1 / Lv.2 / Lv.3 / Lv.4 (required)
  --checkpoint  Model checkpoint (default: data/checkpoints/best.pt)
  --seg_len     Bars per segment (default: 8, range: 4–8)
  --temperature Sampling temperature (default: 1.2; higher = more varied)
  --top_k       Top-k sampling (default: 10; 0 = greedy argmax)
  --device      Device override (default: auto-detect CUDA)
  ```

- Inference improvements over naive greedy decode:
    - **Forced Dtgt** (`init_token_idx`) — the target level token is injected directly into the decoder prefix, preventing the model from predicting the wrong difficulty level as its first token
    - **Top-k sampling** (`top_k=10, temperature=1.2`) — breaks the greedy repetition collapse where the model would output the same note hundreds of times

- Usage:
  ```bash
  python infer.py --input mxl/X/XX/Qm....mxl --output output.mxl --level Lv.1
  python infer.py --input mxl/X/XX/Qm....mxl --output output.mxl --level Lv.1 --temperature 0.8 --top_k 5
  ```

- **Status: DONE**

---

### Phase 5 — Evaluation

**[5.1] `evaluate.py`**
- Compute note density, pitch width, polyphony for generated vs. reference
- Jensen-Shannon divergence between generated and human-level distributions
- Syntax error rate and structure error rate
- **Status: TODO**

---

### Phase 6 — AI Auto-Orchestration & Cross-Instrument Extension (Piano ➔ Duet)

**Motivation & data-source change vs. Section 2.1.**
Section 2.1 concluded that cross-instrument training was infeasible because only **141 songs** in PDMX have both a piano-only arrangement and a violin-only arrangement (1,381 cross-pairs total). Phase 6 sidesteps that bottleneck by **changing the data source**: instead of pairing two separately-uploaded arrangements, we use PDMX's existing **piano + violin duet scores** (MIDI program 0 + 40 inside a single score) and **synthesize the source side ourselves** via reverse augmentation. This converts the problem from "find paired data" to "split data we already have", and unlocks a much larger pool of scores.

**[6.1] Duet Data Processing & Reverse Augmentation (`tokenize_duet.py`)**

- Filter PDMX to scores whose `tracks` column contains both `0` (piano) and `40` (violin). Verified against `PDMX.csv` (254,077 rows total): **1,890 scores match**.
- Composition of those 1,890 scores (important — they are mostly NOT pure duets):

  | Class | Count | Share |
  |---|---:|---:|
  | Contains other instruments (e.g. `0-40-41-42`, chamber / small orchestra) | 1,508 | 79.8% |
  | Pure piano + violin (`0-40` exactly) | 315 | 16.7% |
  | Piano / violin doublings only (e.g. `0-40-40`, `0-0-40`) | 67 | 3.5% |

  Top co-occurring programs alongside piano+violin: viola (41), cello (42), flute (73), trumpet (56) — meaning many of the 1,890 are actually string quartets or small-ensemble arrangements rather than literal duets.

- **Decision: use all 1,890 scores (Option B), discard non-0 / non-40 tracks at preprocessing time.**
  - For each score, pick the **first** program-0 part as the piano, and the **first** program-40 part as the violin melody.
  - For `0-40-40` (two violins): use Violin 1 only.
  - For `0-0-40` and similar (multiple pianos): use the first piano part only.
  - **Drop** all other programs (41, 42, 73, …) entirely.
- Run `MusicXML_to_tokens()` per track to isolate **[Violin Melody]** and **[Piano Accompaniment]** as separate token streams.
- **Actual results (run 2026-05-18):** 1,735 / 1,890 scores tokenized successfully (91.8%). 65.1 MB total output. 151,475 bars across all duets; median 64 bars/score. 155 failures dominated by upstream `score_to_tokens.aggregate_notes()` raising `Cannot insert None into a tag`.
- **Status:** DONE. Code: `Phase06/extract_duet_mxl.py`, `Phase06/tokenize_duet.py`.

**[6.2] Duet Pair Building (`build_pairs_duet.py`)**

- **Dataset A — Melody Extraction:** `[Pseudo Piano Solo] ➔ [Violin Melody]`
- **Dataset B — Auto-Orchestration:** `[Pseudo Piano Solo] ➔ [Original Duet (Violin + Piano)]`
- **Actual results (run 2026-05-23):** 73,366 Dataset A + 73,366 Dataset B = 146,732 segments from 1,733 / 1,735 scores (99.9%).
- **Status:** DONE. Code: `Phase06/build_pairs_duet.py`. Output: `Phase06/pairs_duet.jsonl`.

**[6.3] Vocabulary & Model Update (`build_vocab.py`, `model.py`)**

- Extend `vocab.json` with track tokens (`<track_piano>`, `<track_violin>`) and task tokens (`<task_melody>`, `<task_duet>`).
- Train from scratch with `train_seq2seq.py` on `pairs_duet.jsonl`.
- **Status:** TODO

**[6.4] Duet Inference (`infer_duet.py`)**

- Input: a real piano-solo MXL + a task flag (`--task melody` or `--task duet`).
- **Status:** TODO

**[6.5] Duet Evaluation (`evaluate_duet.py`)**

- Melody extraction: note-level precision / recall / F1 vs. held-out original violin part.
- Auto-orchestration: distributional metrics per track + inter-track interaction metrics.
- **Status:** TODO

---

## 6. File Structure

```
score-rearrangement/
    mxl/                         raw MusicXML files (PDMX)
    tokens/                      tokenized JSON files (output of 1.1)
    data/
        pairs.jsonl              training pairs (output of 1.2)
        score_list.csv           per-score difficulty table (output of list_scores.py)
        vocab.json               vocabulary (output of 1.3)
        checkpoints/             saved model weights
            best.pt              best checkpoint by val loss
            epoch_NNNN.pt        periodic snapshots
            train_log.csv        epoch-by-epoch training log
    score_to_tokens.py           MXL → ST+ tokens
    tokens_to_score.py           ST+ tokens → MXL
    tokenize_all.py              batch tokenization [Phase 1.1]
    build_pairs.py               pair generation + difficulty labeling [Phase 1.2]
    build_vocab.py               vocabulary builder [Phase 1.3]
    audit_pairs.py               sample N unique pairs → CSV for manual quality review
    analyze_pairs.py             full-dataset pair quality analysis → contamination stats
    model.py                     seq2seq Transformer [Phase 2.1]
    dataset_seq2seq.py           PyTorch Dataset [Phase 2.2]
    train_seq2seq.py             training script [Phase 3.1]
    infer.py                     inference script [Phase 4.1]
    evaluate.py                  evaluation metrics [Phase 5.1]
    list_scores.py               generates score_list.csv for test score selection
    PDMX.csv                     PDMX metadata
    ScoreRearrangement-ProjectSpecification.md
```

---

## 7. Key References

1. Suzuki, M. (2023). Piano score rearrangement into multiple difficulty levels via notation-to-notation approach. *EURASIP Journal on Audio, Speech, and Music Processing.* https://doi.org/10.1186/s13636-023-00321-7

2. ScoreRearrangement GitHub (ST+ tokenization tools): https://github.com/suzuqn/ScoreRearrangement

3. PDMX Dataset: Piano Data from MuseScore eXchange
