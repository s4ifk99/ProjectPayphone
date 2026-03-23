# Training Iteration Loop

Improve the Payphone storytelling model through repeated generation and retraining.

## Pipeline

```
dataset (400-500 stories)
    → train model (QLoRA on Colab T4)
    → generate new stories (unseen cases)
    → validate and clean
    → add to dataset
    → retrain (same Colab notebook, fresh upload of JSONL)
```

**Canonical training + merge path:** [TRAINING_COLAB.md](TRAINING_COLAB.md) (train) and [IMPORT_OLLAMA.md](IMPORT_OLLAMA.md) (GGUF). Each retrain iteration: update `training_payphone.jsonl` locally, then re-run `train_payphone_colab.ipynb` in Colab and repeat the import notebook—**not** `train_payphone_model.py` unless you are doing optional local GPU development (see [training/README.md](training/README.md)).

## Steps

### 1. Train initial model

**Local prep:**

```bash
python scripts/validate_training_dataset.py
python scripts/convert_training_dataset.py
```

**Train on Colab T4:** Open `notebooks/train_payphone_colab.ipynb`, upload `training_data_1/training_payphone.jsonl`, run all cells. See [TRAINING_COLAB.md](TRAINING_COLAB.md).

**Merge + GGUF on Colab T4:** Run `notebooks/colab_full_import.ipynb` with the downloaded adapter zip. See [IMPORT_OLLAMA.md](IMPORT_OLLAMA.md).

**Optional test (local, if you have a local HF merge):**

```bash
python scripts/test_payphone_model.py
```

### 2. Generate new stories with fine-tuned model

After exporting the model to Ollama (see [training/README.md](training/README.md) and [IMPORT_OLLAMA.md](IMPORT_OLLAMA.md)):

```bash
# Set OLLAMA_MODEL=payphone-story
# Update scripts/generate_training_stories.py to use payphone-story

# Generate 500 more stories on unseen cases
python scripts/generate_training_stories.py --limit 500
```

Or use the FastAPI app to generate stories interactively.

### 3. Validate and merge

```bash
python scripts/validate_training_dataset.py --input training_data_1/stories_dataset.csv --output training_data_1/stories_dataset_cleaned.csv
```

Merge the new cleaned stories with the existing training set (or replace if starting fresh).

### 4. Retrain

```bash
python scripts/convert_training_dataset.py
```

Then **Colab:** re-upload the updated `training_payphone.jsonl` and run `train_payphone_colab.ipynb` again; run `colab_full_import.ipynb` for the new adapter.

### 5. Repeat

Each iteration produces higher-quality training data, improving the model's consistency for:

- 400-600 word stories
- Dark historical legal fiction tone
- Implicit Hero's Journey structure
- Natural prose (no headings or labels)
