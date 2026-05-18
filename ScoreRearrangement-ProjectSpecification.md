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

### Data quality findings (after initial training)

PDMX "same song" pairs are user-uploaded arrangements that share a song title but may not share the same melody, key, or structure. This caused the first-round model to output piano music that sounded like a different piece entirely rather than a transformed version of the input.

**Compatibility filter added to `build_pairs.py`:**
- Same key signature required
- Same time signature required
- Note density ratio ≤ 3× (one arrangement cannot have 3× more notes/bar than the other)

This filter removes pairs that share only a title. Fewer but higher-quality pairs are expected to produce better melody-preserving transformations.

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
- **Status: DONE**

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

- Compatibility filters (key/time/density matching):
  - Same key signature — different keys almost certainly means different arrangements
  - Same time signature — structurally incompatible otherwise
  - Note density ratio ≤ 3× — if one arrangement has far more notes, they are likely unrelated pieces
- Save as `pairs.jsonl`
- **Status: DONE**

**[1.3] `build_vocab.py`**
- Scan all token files to collect unique tokens
- Add special tokens: `<pad>`, `<sos>`, `<eos>`, `Lv.1`, `Lv.2`, `Lv.3`, `Lv.4`
- Save `vocab.json` (token → index mapping)
- Token strings need to encode to indexes for feeding to model
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

**Round 1 training results (noisy pairs):**
- Trained 60 epochs, val_loss = 1.24, LR decayed to minimum (1e-5)
- Output was musically valid in structure but did not sound like the same song as the input
- Root cause: training pairs matched by song title only, not musical content

**Round 2 plan (after reprocessing with compatibility filter):**
- Run `python build_pairs.py` to regenerate `pairs.jsonl` with key/time/density filter
- Train from scratch: `python train_seq2seq.py --epochs 100 --lr 1e-3`
- Training from scratch (not resuming) is preferred since the data distribution changes significantly

- **Status: Round 1 DONE, Round 2 pending dataset reprocessing**

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
**[6.1] Expand Data Processing & Reverse Augmentation (tokenize_duet.py)**

Extract 1,600+ "Piano + Violin" duet scores (MIDI program 0 and 40) from PDMX.

Implement track splitting to isolate [Violin Melody] and [Piano Accompaniment].

Implement Reverse Data Augmentation: Merge the violin melody tokens into the piano accompaniment track to synthesize complex [Pseudo Piano Solo] scores as the source input.

* Status: TODO (pending Phase 1–5 completion)

**[6.2] Expand Pair Building (build_pairs.py)**

Construct perfectly aligned training pairs utilizing the synthesized data (bypassing the need for external datasets):

Dataset A (Melody Extraction): [Pseudo Piano Solo] ➔ [Violin Melody]

Dataset B (Auto-Orchestration): [Pseudo Piano Solo] ➔ [Original Duet (Violin + Piano)]

* Status: TODO

**[6.3] Update Model & Vocabulary (model.py & vocab.json)**

Update vocab.json with new track-specific tokens (e.g., <track_violin>, <track_piano>) to support multi-track output.

Adjust the Seq2Seq Transformer to handle sequence translation and feature decoupling (Track Allocation).

Train new model weights from scratch for the Orchestration task, learning to automatically expand a single piano solo into a duet.

* Status: TODO

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
