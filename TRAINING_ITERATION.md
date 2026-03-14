# Training Iteration Loop

Improve the Payphone storytelling model through repeated generation and retraining.

## Pipeline

```
dataset (400-500 stories)
    → train model (QLoRA)
    → generate new stories (unseen cases)
    → validate and clean
    → add to dataset
    → retrain
```

## Steps

### 1. Train initial model

```bash
# Validate and convert dataset
python scripts/validate_training_dataset.py
python scripts/convert_training_dataset.py

# Train
python scripts/train_payphone_model.py

# Test
python scripts/test_payphone_model.py
```

### 2. Generate new stories with fine-tuned model

After exporting the model to Ollama (see [training/README.md](training/README.md)):

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
python scripts/train_payphone_model.py
```

### 5. Repeat

Each iteration produces higher-quality training data, improving the model's consistency for:

- 400-600 word stories
- Dark historical legal fiction tone
- Implicit Hero's Journey structure
- Natural prose (no headings or labels)
