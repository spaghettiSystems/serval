import pytorch_lightning as pl
from pytorch_lightning.callbacks import GradientAccumulationScheduler
import numpy as np
import torch

from pytorch_lightning.loggers import WandbLogger
from models.whisper import *

from pytorch_lightning.callbacks import LearningRateMonitor
from whisper_flash_attention import *
import os


from models.main import *
from datasets import *
from create_dataset_backbone_embeddings import PrecomputedDataset, nested_collate_fn
import argparse
from constants import *
import random

torch.set_float32_matmul_precision("medium")


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True


os.environ["WANDB_MODE"] = "offline"


os.environ["WANDB__SERVICE_WAIT"] = "300"


seed = 34
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

if __name__ == "__main__":
    print("Current working directory: {0}".format(os.getcwd()))

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_model",
        type=str,
        default="openai/whisper-large-v3",
        help="Name of the base model backbone",
    )
    parser.add_argument(
        "--dataset_name", type=str, default="EmoDBDataset", help="Name of the dataset"
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU to use")
    parser.add_argument(
        "--prefix_path",
        type=str,
        default="../../../eseg_test_data/",
        help="Prefix path for the dataset folders.",
    )
    args = parser.parse_args()

    base_model = args.base_model
    dataset_name = args.dataset_name
    gpu = args.gpu
    prefix_path = args.prefix_path

    run_name = f"{clean_string(base_model)}-{clean_string(dataset_name)}"

    dataset_path = os.path.join(
        prefix_path, dataset_names_to_folders[dataset_name], "embeddings", run_name
    )
    dataset = PrecomputedDataset(dataset_path, dataset_name)
    sample_input = dataset[0][0]
    input_dim = sample_input.shape[-1]

    schedule = {
        0: 1,
    }

    model = ImprovedMLPHead(input_dim, 8, [512], dropout_rate=0.1)
    model = EncoderPL(model, schedule)

    dataset = SERDataModule(
        dataset,
        base_model,
        batch_size=32,
        custom_collate=nested_collate_fn,
        ood_tests=True,
        prefix=prefix_path,
    )

    accumulator = GradientAccumulationScheduler(scheduling=schedule)
    lr_monitor = LearningRateMonitor(logging_interval="step")
    wandb_logger = WandbLogger(
        project="ser_evals",
        name=run_name,
        offline=True,
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[gpu],
        precision="32-true",
        enable_checkpointing=True,
        gradient_clip_val=0.5,
        max_epochs=100,
        check_val_every_n_epoch=10,
        num_sanity_val_steps=2,
        default_root_dir="./checkpoints/",
        log_every_n_steps=1,
        callbacks=[
            lr_monitor,
            accumulator,
        ],
        logger=wandb_logger,
    )

    trainer.fit(model, dataset)
    trainer.validate(model, dataset)
