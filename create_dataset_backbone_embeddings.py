import os
import argparse
import torch
from tqdm import tqdm
from transformers import AutoProcessor, Wav2Vec2Processor
from cleaner_datasets import *
from pathlib import Path
from constants import *
import einops


def collate_fn(batch):
    input_features, labels, indices = zip(*batch)

    if all(x.size() == input_features[0].size() for x in input_features):
        input_features = torch.stack(input_features)
        attention_mask = None
    else:
        max_length = max(x.size(-1) for x in input_features)
        padded_features = []
        attention_mask = []

        for features in input_features:
            padding = torch.zeros(features.size(0), max_length - features.size(-1))
            padded_features.append(torch.cat((features, padding), dim=-1))

            mask = torch.ones(features.size(-1))
            mask_padding = torch.zeros(max_length - features.size(-1))
            attention_mask.append(torch.cat((mask, mask_padding), dim=-1))

        input_features = torch.stack(padded_features)
        attention_mask = torch.stack(attention_mask)
    input_features = input_features.squeeze(1)

    labels = torch.stack(labels)
    indices = torch.stack(indices)

    return input_features, labels, indices, attention_mask


def smart_collate_fn(batch):
    input_features, labels, indices = zip(*batch)

    channels_dim = None
    for dim in range(input_features[0].ndim):
        if all(x.size(dim) == input_features[0].size(dim) for x in input_features):
            channels_dim = dim
            break

    if channels_dim is None:
        raise ValueError("Unable to determine the channels dimension.")

    is_spectrogram = input_features[0].size(channels_dim) > 1

    seq_len_dim = 0 if channels_dim == 1 else 1

    if all(
        x.size(seq_len_dim) == input_features[0].size(seq_len_dim)
        for x in input_features
    ):
        input_features = torch.stack(input_features)
        attention_mask = None
    else:
        max_length = max(x.size(seq_len_dim) for x in input_features)
        padded_features = []
        attention_mask = []

        for features in input_features:
            padding = torch.zeros(
                *features.size()[:seq_len_dim],
                max_length - features.size(seq_len_dim),
                *features.size()[seq_len_dim + 1 :],
            )
            padded_features.append(torch.cat((features, padding), dim=seq_len_dim))

            mask = torch.ones(features.size(seq_len_dim))
            mask_padding = torch.zeros(max_length - features.size(seq_len_dim))
            attention_mask.append(torch.cat((mask, mask_padding), dim=0))

        input_features = torch.stack(padded_features)
        attention_mask = torch.stack(attention_mask)

    labels = torch.stack(labels)
    indices = torch.stack(indices)

    return (
        input_features,
        labels,
        indices,
        attention_mask,
        is_spectrogram,
        channels_dim,
        seq_len_dim,
    )


class PrecomputedDataset(torch.utils.data.Dataset):
    def __init__(self, path, dataset_name=None):
        self.embeddings = list(Path(path).rglob("*.pt"))
        self.embeddings.sort(key=lambda x: int(x.stem.split("_")[0]))
        self.dataset_name = dataset_name

        self.path = path

        if len(self.embeddings) == 0:
            raise ValueError(f"No embeddings found in {path}")

    def __getitem__(self, idx):
        embedding = torch.load(self.embeddings[idx], map_location="cpu")
        output, label, index = embedding

        if "clap" in self.path:
            output = einops.rearrange(output, "e a b -> (a b) e")

        return output, label, index

    def __len__(self):
        return len(self.embeddings)

    def get_dataset_distribution(self):
        if self.dataset_name is not None and self.dataset_name in DATASET_DISTRIBUTIONS:
            return DATASET_DISTRIBUTIONS[self.dataset_name]
        else:
            raise ValueError(f"Dataset distribution not found for {self.dataset_name}")


def nested_collate_fn(batch):
    input_features, labels, indices = zip(*batch)

    labels = torch.stack(labels)
    indices = torch.stack(indices)
    return input_features, labels, indices


def main(base_model, data_dirs, embedding_dir, device, batch_size, num_workers):
    model_config = BASE_MODEL_CONFIG[base_model]
    model_class = model_config["model_class"]
    processor_class = model_config["processor_class"]

    try:
        model = model_class.from_pretrained(base_model).to(device)
    except:
        model = model_class.from_pretrained(base_model, trust_remote_code=True).to(
            device
        )
    processor = processor_class.from_pretrained(base_model)

    if base_model.startswith("openai/whisper"):
        model = model.get_encoder()
    if "clap" in base_model:
        model = model.audio_model

        model.old_forward = model.forward
        model.forward = lambda input_features, attention_mask=None: model.old_forward(
            input_features,
        )

    for dataset_class, data_dir in data_dirs.items():
        print(f"Processing {base_model} with {dataset_class}")

        dataset = globals()[dataset_class](data_dir, processor, downsample_factor=1)

        dataloader = torch.utils.data.DataLoader(
            dataset,
            shuffle=False,
            collate_fn=smart_collate_fn,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        def clean_string(string):
            return "".join(e for e in string if e.isalnum()).lower()

        file_base_model = clean_string(base_model)
        file_dataset_class = clean_string(dataset_class)
        save_dir = os.path.join(
            data_dir, embedding_dir, f"{file_base_model}-{file_dataset_class}"
        )
        os.makedirs(save_dir, exist_ok=True)

        progress_bar = tqdm(
            total=len(dataloader), desc=f"Processing {base_model} with {dataset_class}"
        )

        for i, batch in enumerate(dataloader):
            (
                inputs,
                labels,
                indices,
                attention_mask,
                is_spectrogram,
                channels_dim,
                seq_len_dim,
            ) = batch
            inputs, labels, indices, attention_mask = (
                inputs.to(device),
                labels.to(device),
                indices.to(device),
                attention_mask.to(device) if attention_mask is not None else None,
            )

            batch_dim = 0
            channels_dim += 1
            seq_len_dim += 1

            with torch.no_grad():
                outputs = get_features(
                    model, inputs.squeeze(1), attn_mask=attention_mask
                )

            input_seq_len = torch.max(indices[:, 0, 1])
            output_seq_len = outputs.size(1)

            empirical_downsampling_factor = input_seq_len // output_seq_len
            chosen_downsampling_factor = model_config["downsampling_factor"]

            effective_length = input_seq_len // chosen_downsampling_factor

            if effective_length != output_seq_len:
                if effective_length != output_seq_len + 1:
                    print(
                        f"Empirical downsampling factor for {base_model} with {dataset_class} does not match the chosen downsampling factor. Observed {empirical_downsampling_factor}, expected {chosen_downsampling_factor}."
                    )

            def clean_up_tensors(tensor):
                return torch.from_numpy(tensor.cpu().contiguous().float().numpy())

            for file_idx, (output, label, index) in enumerate(
                zip(outputs, labels, indices)
            ):
                global_dataset_idx = i * batch_size + file_idx
                index = index[0]
                start_idx = index[0] // chosen_downsampling_factor
                end_idx = index[1] // chosen_downsampling_factor
                output = (
                    output if "clap" in base_model else output[start_idx:end_idx, :]
                )

                output, label, index = (
                    clean_up_tensors(output),
                    clean_up_tensors(label),
                    clean_up_tensors(index),
                )

                torch.save(
                    (output, label, index),
                    os.path.join(save_dir, f"{global_dataset_idx}_embd.pt"),
                )

            progress_bar.update(1)

        progress_bar.close()

        precomputed_dataset = PrecomputedDataset(save_dir)
        print(f"Finished processing {base_model} with {dataset_class}")


def get_features(model, inputs, attn_mask=None):
    if hasattr(model.config, "input_name"):
        input_name = model.config.input_name
        return model(
            **{input_name: inputs, "attention_mask": attn_mask}
        ).last_hidden_state
    else:
        try:
            input_name = "input_values"
            return model(
                **{input_name: inputs, "attention_mask": attn_mask}
            ).last_hidden_state
        except:
            input_name = "input_features"
            return model(
                **{input_name: inputs, "attention_mask": attn_mask}
            ).last_hidden_state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Precompute embeddings for datasets using specified base models."
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="facebook/wav2vec2-large-robust",
        help="Name of the base model to use for embedding computation.",
    )
    parser.add_argument(
        "--embedding_dir",
        type=str,
        default="embeddings",
        help="Directory to store the computed embeddings.",
    )
    parser.add_argument(
        "--device", type=int, default=0, help="GPU device index to use for computation."
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for processing."
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of workers for data loading."
    )
    parser.add_argument(
        "--prefix_path",
        type=str,
        default="../../../eseg_test_data/",
        help="Prefix path for the dataset folders.",
    )
    args = parser.parse_args()

    for dataset_name, folder in dataset_names_to_folders.items():
        dataset_names_to_folders[dataset_name] = os.path.join(args.prefix_path, folder)

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    main(
        args.base_model,
        dataset_names_to_folders,
        args.embedding_dir,
        device,
        args.batch_size,
        args.num_workers,
    )
