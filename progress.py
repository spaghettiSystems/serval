import os
import argparse
from tqdm import tqdm
from constants import *


def count_files(directory, extension):
    count = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(extension):
                count += 1
    return count


def check_tokenization_progress(prefix_path):
    for dataset_name, folder in tqdm(dataset_names_to_folders.items(), desc="Datasets"):
        dataset_path = os.path.join(prefix_path, folder)
        wav_count = count_files(dataset_path, ".wav")

        print(f"Dataset: {dataset_name}")
        print(f"Total .wav files: {wav_count}")

        for model_name in tqdm(BASE_MODEL_CONFIG, desc="Models", leave=False):
            embeddings_folder = os.path.join(
                dataset_path,
                "embeddings",
                f"{clean_string(model_name)}-{clean_string(dataset_name)}",
            )

            if not os.path.exists(embeddings_folder):
                print(f"Embeddings folder not found for model: {model_name}")
                continue

            pt_count = count_files(embeddings_folder, ".pt")

            completion_percentage = (
                (pt_count / wav_count) * 100 if wav_count > 0 else 100
            )

            print(f"Model: {model_name}")
            print(f"Tokenized .pt files: {pt_count}")
            print(f"Completion percentage: {completion_percentage:.2f}%")
            print()

        print("-" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check tokenization progress for datasets and models."
    )
    parser.add_argument(
        "--prefix_path",
        type=str,
        default="../../../eseg_test_data/",
        help="Prefix path for the dataset folders.",
    )
    args = parser.parse_args()

    check_tokenization_progress(args.prefix_path)
