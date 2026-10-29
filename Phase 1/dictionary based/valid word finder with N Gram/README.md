# Valid Word Finder

This folder contains a small experimentation workspace for validating whether a
word should be treated as a real or acceptable English-like word. The project
contains multiple validators, comparison scripts, source datasets, generated
result spreadsheets, and diagnostic plots.

The codebase is centered around a few different ideas:

- lexical validation using `wordfreq`
- character-level backoff language modeling
- character-based pronounceability
- syllable-level pronounceability
- phoneme-sequence wordlikeness
- orthographic phonotactic scoring
- hybrid models that combine lexical familiarity with pronounceability

## Folder Purpose

The purpose of this folder is to:

1. train different word-validity or wordlikeness models
2. test them on labeled Excel sheets such as `special test.xlsx` and
   `special test 2.xlsx`
3. export per-word scores to spreadsheets
4. visualize score distributions with simple 1D plots
5. compare older character-based hybrid logic with newer syllable-based hybrid logic

## Current Layout

```text
valid word finder/
├── README.md
├── character_backoff_language_model.py
├── character_backoff_pronounceability_validator.py
├── export_hybrid_validation_results.py
├── hybrid_lexical_character_pronounceability_validator.py
├── hybrid_lexical_pronounceability_validator.py
├── lexical_frequency_validator.py
├── orthographic_phonotactic_validator.py
├── phoneme_sequence_wordlikeness_validator.py
├── plot_score_distributions.py
├── syllable_backoff_pronounceability_validator.py
├── special_test_2_hybrid_comparison.xlsx
├── datasets/
├── results/
└── __pycache__/
```

## Important Note About The Root Folder

Most non-code files were moved into `datasets/` or `results/`.

One file is still in the root:

- `special_test_2_hybrid_comparison.xlsx`

That file remained in the root because Windows reported it was open in another
process and would not allow it to be moved at the time of cleanup. Once it is
closed, it can be moved into `results/spreadsheets/`.

## Main Workflows

### 1. Run the older character-based hybrid

```powershell
python hybrid_lexical_character_pronounceability_validator.py --word apple
python hybrid_lexical_character_pronounceability_validator.py --eval-file "datasets/special test 2.xlsx"
```

### 2. Run the newer syllable-based hybrid

```powershell
python hybrid_lexical_pronounceability_validator.py --word apple
python hybrid_lexical_pronounceability_validator.py --eval-file "datasets/special test 2.xlsx"
```

### 3. Export hybrid predictions to Excel

```powershell
python export_hybrid_validation_results.py --input-file "datasets/special test.xlsx" --output-file "results/spreadsheets/special_test_hybrid_results.xlsx"
```

### 4. Plot score distributions

```powershell
python plot_score_distributions.py --input-file "results/spreadsheets/special_test_hybrid_results.xlsx" --output-dir "results/plots"
```

## Python Files

### `character_backoff_language_model.py`

Purpose:
- Provides the basic character-level backoff chain rule language model.

What it does:
- Loads words and counts from Excel.
- Learns which characters tend to follow which earlier character contexts.
- Backs off to shorter contexts when an exact longer context is missing.
- Can score a single word or evaluate a dataset.

Why it matters:
- It is the lowest-level character modeling utility used by character-based
  validators.

Typical input:
- `datasets/train_word_count.xlsx`

Typical output:
- probability values or evaluation summary printed to the terminal

### `character_backoff_pronounceability_validator.py`

Purpose:
- Implements the older character-based pronounceability model.

What it does:
- Uses the character backoff model from
  `character_backoff_language_model.py`.
- Splits words into alternating vowel and consonant chunks.
- Uses both character sequence plausibility and chunk plausibility.
- Produces a pronounceability score and valid/invalid decision.

Why it matters:
- This is the older pronounceability component used by the old hybrid model.

### `syllable_backoff_pronounceability_validator.py`

Purpose:
- Implements the newer syllable-level pronounceability model.

What it does:
- Converts words to phonemes using `cmudict` and `g2p_en`.
- Splits phoneme sequences into syllable-like groups.
- Converts those syllables into tokens.
- Trains a backoff model over syllable tokens.
- Scores new words by syllable-sequence plausibility.

Why it matters:
- This is the pronounceability component used by the current syllable-based hybrid.

### `lexical_frequency_validator.py`

Purpose:
- Implements pure lexicon-based validation using `wordfreq`.

What it does:
- Builds a lexicon of known English words.
- Assigns each word a Zipf frequency score.
- Marks an input word as valid if it is present in the lexicon.
- Returns both raw Zipf and normalized familiarity scores.

Why it matters:
- This is the lexical half of the hybrid validators.

### `hybrid_lexical_character_pronounceability_validator.py`

Purpose:
- Implements the old hybrid model: lexical familiarity plus character-based pronounceability.

What it does:
- If a word is in the lexicon, combines:
  - normalized lexical familiarity
  - character-based pronounceability
  - a small common-word bonus
- If a word is not in the lexicon, falls back to pronounceability alone.

Why it matters:
- This preserves the older hybrid approach so it can be compared to the newer syllable-based hybrid.

### `hybrid_lexical_pronounceability_validator.py`

Purpose:
- Implements the current hybrid model: lexical familiarity plus syllable-based pronounceability.

What it does:
- Uses the same hybrid combination structure as the old hybrid.
- Replaces character-based pronounceability with syllable-based pronounceability.

Why it matters:
- This is the main current hybrid file when testing the syllable-level approach.

Important clarification:
- Even though the name does not explicitly say `syllable`, this file currently
  uses `syllable_backoff_pronounceability_validator.py`.

### `phoneme_sequence_wordlikeness_validator.py`

Purpose:
- Implements a phoneme-sequence wordlikeness model.

What it does:
- Converts words to phonemes.
- Builds phoneme n-gram statistics.
- Uses shape information such as onset/coda style structure.
- Adds a small lexical familiarity bonus.

Why it matters:
- This is a sound-based alternative to character-only or lexicon-only validation.

### `orthographic_phonotactic_validator.py`

Purpose:
- Implements a spelling-pattern-based phonotactic or orthographic wordlikeness model.

What it does:
- Uses character n-grams.
- Uses substring chunk statistics.
- Uses simple vowel/consonant run heuristics.
- Adds lexical familiarity.

Why it matters:
- This is useful when you want English-looking spelling patterns without going all the way to phoneme modeling.

### `export_hybrid_validation_results.py`

Purpose:
- Writes hybrid model outputs into Excel sheets.

What it does:
- Reads an input sheet.
- Scores every word with the hybrid validator.
- Writes columns such as:
  - word
  - count
  - hybrid score
  - predicted validity
  - predicted label
  - ground truth fields when available
  - zipf score
  - pronounceability score

Why it matters:
- This is the main reporting helper used to inspect model outputs in spreadsheet form.

### `plot_score_distributions.py`

Purpose:
- Draws simple 1D plots for hybrid result spreadsheets.

What it does:
- Reads exported result sheets.
- Produces one plot per score column.
- Colors dots by ground truth when available, otherwise by predicted label.
- Saves plots under `results/plots/`.

Why it matters:
- This is the easiest visual diagnostic tool in the folder.

## Dataset Files

All dataset files now live in `datasets/`.

### `datasets/train_word_count.xlsx`

Purpose:
- Main training set for the pronounceability and backoff models.

Used by:
- `character_backoff_language_model.py`
- `character_backoff_pronounceability_validator.py`
- `syllable_backoff_pronounceability_validator.py`
- both hybrid validators
- other validator scripts that learn from your word-count training data

Expected columns:
- `word`
- `count`

### `datasets/val_word_count.xlsx`

Purpose:
- Validation set of weighted words.

Likely use:
- external evaluation or future threshold tuning

Expected columns:
- `word`
- `count`

### `datasets/test_word_count.xlsx`

Purpose:
- Unlabeled test-style word-count sheet.

Used for:
- scoring large numbers of words
- exporting hybrid predictions
- plotting predicted score distributions

Expected columns:
- `word`
- `count`

### `datasets/special test.xlsx`

Purpose:
- Small labeled benchmark sheet.

Used for:
- manual spot checks
- score exports
- visualization

Expected columns:
- `word`
- `label`

### `datasets/special test 2.xlsx`

Purpose:
- Larger labeled benchmark sheet.

Used for:
- comparing old and new hybrid variants
- broader evaluation than the smaller special test sheet

Expected columns:
- `word`
- `label`

### `datasets/SENT_based_split.xlsx`

Purpose:
- A larger workbook already present in the project.

Current status:
- It exists in the dataset folder, but it is not part of the core workflow used
  in the recent hybrid-validator tests and exports.

### `datasets/wordfreq_english_lexicon.csv`

Purpose:
- Local CSV resource related to lexical frequency information.

Current status:
- The lexical validator mainly builds its lexicon through the `wordfreq`
  package directly, but this CSV is still stored here as a project data asset.

## Results Spreadsheets

All generated result workbooks are stored in `results/spreadsheets/`, except for
the one locked workbook still in the root.

### `results/spreadsheets/special_test_hybrid_results.xlsx`

Purpose:
- Per-word hybrid output exported from `datasets/special test.xlsx`

Contains:
- word
- score
- predicted validity
- ground truth
- lexical and pronounceability detail columns

### `results/spreadsheets/special_test_2_hybrid_results.xlsx`

Purpose:
- Per-word hybrid output exported from `datasets/special test 2.xlsx`

### `results/spreadsheets/test_word_count_hybrid_results.xlsx`

Purpose:
- Per-word hybrid output exported from `datasets/test_word_count.xlsx`

Note:
- This sheet does not have ground-truth labels because the source sheet is unlabeled.

### `results/spreadsheets/special_test_scored.xlsx`

Purpose:
- An earlier multi-model scoring sheet created during comparison work.

### `results/spreadsheets/special_test_2_char_vs_syllable_hybrid_comparison.xlsx`

Purpose:
- Direct comparison of the old char+lexical hybrid vs the new syllable+lexical hybrid on `special test 2`.

### `results/spreadsheets/_rename_check.xlsx`

Purpose:
- Temporary validation output created while checking that the renamed scripts still worked.

### `special_test_2_hybrid_comparison.xlsx`

Purpose:
- Another comparison workbook from earlier testing.

Current status:
- Still in the root because it was locked by another process during folder cleanup.

Recommended destination:
- `results/spreadsheets/special_test_2_hybrid_comparison.xlsx`

## Plot Files

All current generated plot images live in `results/plots/`.

### `results/plots/hybrid_score_1d_plot.png`

Purpose:
- 1D plot for hybrid scores from the main small special test export.

### `results/plots/pronounceability_score_1d_plot.png`

Purpose:
- 1D plot for pronounceability scores from the main small special test export.

### `results/plots/zipf_score_1d_plot.png`

Purpose:
- 1D plot for lexical Zipf scores from the main small special test export.

### `results/plots/special_test_2/`

Purpose:
- Contains plots generated from `special_test_2_hybrid_results.xlsx`

Expected files:
- `special_test_2_hybrid_score_1d_plot.png`
- `special_test_2_pronounceability_score_1d_plot.png`
- `special_test_2_zipf_score_1d_plot.png`

### `results/plots/test_word_count/`

Purpose:
- Contains plots generated from `test_word_count_hybrid_results.xlsx`

Expected files:
- `test_word_count_hybrid_score_1d_plot.png`
- `test_word_count_pronounceability_score_1d_plot.png`
- `test_word_count_zipf_score_1d_plot.png`

## `__pycache__/`

Purpose:
- Auto-generated Python bytecode cache files.

Meaning:
- These are not source files and usually do not need manual editing.

## Relationship Between The Main Models

The most important code relationship is:

- `character_backoff_language_model.py`
  -> base character sequence model
- `character_backoff_pronounceability_validator.py`
  -> old pronounceability built on the character model
- `syllable_backoff_pronounceability_validator.py`
  -> new pronounceability built on syllable token sequences
- `lexical_frequency_validator.py`
  -> lexical familiarity / dictionary-style scoring
- `hybrid_lexical_character_pronounceability_validator.py`
  -> old hybrid = lexical + character pronounceability
- `hybrid_lexical_pronounceability_validator.py`
  -> new hybrid = lexical + syllable pronounceability

## Which File To Use For What

Use:

- `hybrid_lexical_character_pronounceability_validator.py`
  if you want the older character-based hybrid
- `hybrid_lexical_pronounceability_validator.py`
  if you want the newer syllable-based hybrid
- `export_hybrid_validation_results.py`
  if you want per-word Excel outputs
- `plot_score_distributions.py`
  if you want plots from those Excel outputs

## Notes About IDE Tabs

Your IDE may still show older tab names such as:

- `hybrid_word_validator.py`
- `export_hybrid_results.py`
- `phoneme_wordlikeness_validator.py`

Those are stale tab references from before the cleanup and rename process. The
current files are the renamed versions documented in this README.

## Recommended Next Cleanup

If you want the folder even cleaner, the next sensible steps would be:

1. close and move `special_test_2_hybrid_comparison.xlsx` into `results/spreadsheets/`
2. optionally rename `hybrid_lexical_pronounceability_validator.py` to
   `hybrid_lexical_syllable_pronounceability_validator.py` for maximum naming clarity
3. add a single comparison/export script that can choose char-hybrid or
   syllable-hybrid through a command-line flag

