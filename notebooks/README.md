# Notebooks – Colab T4 workflow

Use these on **Google Colab** with **Runtime → Change runtime type → T4 GPU**. Step-by-step checklist: **[TRAINING_COLAB.md](../TRAINING_COLAB.md#colab-t4-checklist)**.

| Notebook / script | Purpose | Documentation |
|--------------------|---------|----------------|
| [train_payphone_colab.ipynb](train_payphone_colab.ipynb) | QLoRA fine-tune; upload `training_payphone.jsonl`; download `payphone-storyteller-lora.zip` | [TRAINING_COLAB.md](../TRAINING_COLAB.md) |
| [colab_training_script.py](colab_training_script.py) | Same training flow as cells to paste into a fresh Colab notebook | [TRAINING_COLAB.md](../TRAINING_COLAB.md) |
| [colab_full_import.ipynb](colab_full_import.ipynb) | Merge LoRA + export GGUF; upload adapter zip; download `payphone-story.gguf` | [IMPORT_OLLAMA.md](../IMPORT_OLLAMA.md) |
| [colab_merge_lora.py](colab_merge_lora.py) | Cell-by-cell copy of `colab_full_import.ipynb` | [IMPORT_OLLAMA.md](../IMPORT_OLLAMA.md) |
| [kaggle_full_import.ipynb](kaggle_full_import.ipynb) | **Kaggle:** merge LoRA → HF checkpoint → download `payphone-merged-hf.zip` (GGUF via Colab or local `convert_to_gguf.sh`) | [IMPORT_OLLAMA.md](../IMPORT_OLLAMA.md) appendix |

**Open from GitHub (example):** replace owner/repo if yours differs.

- Training: `https://colab.research.google.com/github/s4ifk99/ProjectPayphone/blob/main/notebooks/train_payphone_colab.ipynb`
- Import: `https://colab.research.google.com/github/s4ifk99/ProjectPayphone/blob/main/notebooks/colab_full_import.ipynb`
