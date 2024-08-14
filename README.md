# **SER Evals**

![SER Evals Banner](figs/very_important.png)

**SER Evals** is a comprehensive repository designed for evaluating Speech Emotion Recognition (SER) models across various datasets and pre-trained models. It automates the entire workflow from dataset preparation, feature extraction, model training, to performance evaluation. The codebase is built to be easy to extend to additional datasets and models, ensuring reproducibility and scalability for SER research. **Arxiv paper coming soon**.

## **Table of Contents**

- [Usage](#usage)
  - [1. Dataset Preparation](#1-dataset-preparation)
  - [2. Embedding Computation](#2-embedding-computation)
  - [3. Model Training and Evaluation](#3-model-training-and-evaluation)
  - [4. Experiment Management](#4-experiment-management)
  - [5. Result Analysis](#5-result-analysis)
  - [6. Progress Monitoring](#6-progress-monitoring)
- [Repository Structure](#repository-structure)
- [License](#license)

## **Usage**

### **1. Dataset Preparation**

SER Evals supports a wide range of datasets, each of which requires specific preprocessing and loading. The datasets are defined in two primary modules:

- **`cleaner_datasets.py`**: Contains classes for various datasets used in the project.

**Steps:**
- Define your dataset paths in the `constants.py` file under `dataset_names_to_folders`. This is the path to your dataset from some root path.
- Load and preprocess datasets using the provided classes in `cleaner_datasets.py`. Most datasets should work out of the box but datasets with annotations in .csv need to be converted to a suitable format.

### **2. Embedding Computation**

Before training, embeddings need to be computed from the raw audio files using pre-trained models.

**Steps:**
- Use `create_dataset_backbone_embeddings.py` to generate embeddings.
- This script processes each dataset using the specified base models and saves the embeddings in the corresponding directories. It needs to be run for every base model.

### **3. Model Training and Evaluation**

Once the embeddings are generated, you can train models on these embeddings using `trainer_wandb.py`. This is the entry point for the training code.

Managing and running multiple experiments simultaneously across GPUs is handled by `run_everything.py`. This script schedules jobs, monitors GPU memory usage, and retries failed experiments automatically. It is highly configurable, allowing you to set the number of GPUs, max concurrent jobs, and memory limits.

### **5. Result Analysis**

After training, use `grab_results2.py` to process log files and extract relevant metrics. This script reads the logs, extracts metrics such as accuracy, F1-score, etc., and organizes them into CSV files for easy analysis. You can find an example execution result in this repository.

### **6. Progress Monitoring**

Use `progress.py` to monitor the progress of embedding computation for each dataset and model combination if desired. This script calculates the percentage of processed files and helps track the completion status of the embedding phase.

## **Repository Structure**

- **`cleaner_datasets.py`**: Dataset definitions and preprocessing for various SER datasets.
- **`create_dataset_backbone_embeddings.py`**: Script for computing embeddings from audio datasets using pre-trained models.
- **`grab_results2.py`**: Processes logs to extract and save model performance metrics.
- **`trainer_wandb.py`**: Manages the training and validation of models using PyTorch Lightning.
- **`run_everything.py`**: Orchestrates and manages the execution of experiments across multiple GPUs.
- **`progress.py`**: Monitors the progress of embedding computation.
- **`constants.py`**: Contains global constants, configurations, and paths used across the repository.

## **License**

This project is licensed under the GNU General Public License v3.0. See the [LICENSE](LICENSE) file for details.