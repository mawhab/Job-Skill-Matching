# TalentCLEF 2025 Job-Skill Matching

This repository contains code for the TalentCLEF 2025 Task B challenge - matching job titles with relevant skills using transformer-based approaches. The system uses various pre-trained language models with different description augmentation strategies.

## Project Structure

```
repo/
├── common/
│   ├── dataset.py          # Dataset implementations
│   ├── defaults.py         # Configuration and templates
│   ├── evaluation_utils.py # Evaluation metrics
│   ├── model_utils.py      # Model handling utilities
│   └── preprocessing_utils.py # Data preprocessing
├── data/
│   ├── esco_data/         # ESCO occupation/skill data
│   ├── task_data/         # TalentCLEF task data
│   └── generated_data/    # LLM-generated descriptions
├── train.py               # Main training script
├── generate_descriptions.py # Description generation script
└── requirements.txt
```

## Setup

1. Create and activate a new conda environment:
```bash
conda create -n talentclef python=3.11
conda activate talentclef
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Clone and setup the TalentCLEF evaluation script:
```bash
git clone https://github.com/TalentCLEF/talentclef25_evaluation_script.git
cd talentclef25_evaluation_script
pip install -r requirements.txt
cd ..
```

## Description Generation

The `generate_descriptions.py` script uses LLaMA 3.1 to generate job descriptions for training and validation sets.

```bash
python generate_descriptions.py --distribution [train/val] --batch_size 32
```

Arguments:
- `--distribution`: Choose between 'train' or 'val' job distributions
- `--batch_size`: Batch size for description generation (default: 32)

Generated descriptions are saved to:
- `data/generated_data/train_descriptions.json`
- `data/generated_data/val_descriptions.json`

## Training

To train a model, use `train.py`. The script supports different encoder models and description augmentation strategies.

```bash
python train.py --model_name e5_instruct --augmentation esco
```

### Training Arguments

- `--batch_size`: Training batch size (default: 4)
- `--epochs`: Number of training epochs (default: 20)
- `--lr`: Learning rate (default: 1e-5)
- `--warmup`: Warmup ratio for scheduler (default: 0.06)
- `--model_name`: Choice of encoder model:
  - `escoxlm_r`: ESCO-XLM-RoBERTa
  - `e5_large`: E5 Large
  - `e5_instruct`: E5 Large Instruct (default)
- `--augmentation`: Description augmentation type:
  - `esco`: ESCO occupational descriptions
  - `llm`: LLM-generated descriptions (default)
  - `no_desc`: No descriptions
- `--checkpoint`: Path to checkpoint for resuming training (optional)

## Data Requirements

The project expects the following data structure:
```
data/
├── task_data/
│   ├── training/
│   │   ├── job2skill.tsv
│   │   ├── jobid2terms.json
│   │   └── skillid2terms.json
│   └── validation/
│       ├── queries
│       ├── corpus_elements
│       └── qrels.txt
├── esco_data/
│   ├── occupations_en.csv
│   └── skills_en.csv
└── generated_data/
    ├── train_descriptions.json
    └── val_descriptions.json
```

This can be modified by changing the paths in common/defaults.py

The data used in this project was obtained from the TalentCLEF 2025 competition dataset and the ESCO (European Skills, Competences, Qualifications and Occupations) database.
