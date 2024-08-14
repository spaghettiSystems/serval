from whisper_flash_attention import *
import torchmetrics

import pytorch_lightning as pl


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os


from constants import *
from create_dataset_backbone_embeddings import PrecomputedDataset


def collate_fn(batch):
    waveforms, labels, indices, _ = zip(*batch)

    padded_waveforms = torch.stack(waveforms)[:, 0, :, :]

    return padded_waveforms, labels, indices, None


class SERDataModule(pl.LightningDataModule):
    """
    2 modes
    1- pass a single dataset to do a linear probe (train on it)
    2- pass a list of datasets to split the first according to split ratio, train on index 0 and eval on the rest
    """

    def __init__(
        self,
        dataset,
        base_model,
        batch_size: int = 32,
        split_ratio=0.8,
        ood_tests=True,
        prefix=None,
        custom_collate=None,
    ):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.collate_fn = custom_collate if custom_collate is not None else collate_fn

        if isinstance(self.dataset, list):
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                self.dataset[0],
                [
                    int(len(self.dataset[0]) * split_ratio),
                    len(self.dataset[0]) - int(len(self.dataset[0]) * split_ratio),
                ],
            )
            self.dataset[0] = self.train_dataset
            self.dataset.insert(1, self.val_dataset)
        else:
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                self.dataset,
                [
                    int(len(self.dataset) * split_ratio),
                    len(self.dataset) - int(len(self.dataset) * split_ratio),
                ],
            )
            self.dataset = [self.train_dataset, self.val_dataset]

            if ood_tests:
                dataset_name = self.dataset[0].dataset.dataset_name
                for cluster, cluster_data in DATASET_CLUSTERS.items():
                    if dataset_name in cluster_data["datasets"]:
                        for ood_dataset_name in cluster_data["datasets"]:
                            if ood_dataset_name != dataset_name:
                                ood_dataset_path = os.path.join(
                                    prefix,
                                    dataset_names_to_folders[ood_dataset_name],
                                    "embeddings",
                                    f"{clean_string(base_model)}-{clean_string(ood_dataset_name)}",
                                )
                                ood_dataset = PrecomputedDataset(
                                    ood_dataset_path, ood_dataset_name
                                )
                                self.dataset.append(ood_dataset)

                                print(f"Added OOD dataset: {ood_dataset_name}")
                        break

        print(len(self.dataset))

    def train_dataloader(self):
        if isinstance(self.dataset, list):
            return self.dataloader_with_config(self.dataset[0])
        return self.dataloader_with_config(self.train_dataset)

    def val_dataloader(self):
        if isinstance(self.dataset, list):
            return [
                self.dataloader_with_config(dataset, False)
                for dataset in self.dataset[1:]
            ]
        return self.dataloader_with_config(self.val_dataset, False)

    def dataloader_with_config(self, dataset, train=True):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=4,
            shuffle=train,
            collate_fn=self.collate_fn,
            prefetch_factor=2,
            pin_memory=True,
            drop_last=train,
            persistent_workers=True,
        )


class EncoderPL(pl.LightningModule):
    def __init__(self, model, accumulation_schedule):
        super().__init__()
        self.model = model

        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        self.batch_size = 32
        self.accumulation_schedule = accumulation_schedule

        self.create_validation_metrics()

        self.current_dataloader_idx = 0

        self.dataloader_end_idx = 0

        self.prefix = None
        self.execute_once = True

    def forward(
        self, input, labels=None, indices=None, train=True, dataloader_idx=None
    ):
        outputs = self.model(
            input,
        )
        loss = 0
        if labels is not None and indices is not None:
            loss = self.custom_loss_function(
                outputs,
                labels,
                indices,
                train,
            )

        return loss, outputs

    def nested_mean(self, nested_tensor, dim):
        tensors = nested_tensor.unbind()

        mean_tensors = [t.mean(dim=dim) for t in tensors]

        result = torch.stack(mean_tensors).squeeze(1)
        return result

    def custom_loss_function(self, logits, labels, indices, train=True, features=None):
        pooled_output = self.nested_mean(logits, dim=0)

        mean_loss = self.criterion(pooled_output[:, :8], labels)

        if not train:
            self.compute_validation_metrics(indices, features, pooled_output, labels)

        return mean_loss

    def training_step(self, batch, batch_idx):
        input_features, selected_labels, indices = batch

        if isinstance(input_features, list):
            input_features = torch.nested.nested_tensor(input_features)
        loss, outputs = self(
            input_features,
            selected_labels,
            indices,
            train=True,
        )
        return {
            "loss": loss,
            "unscaled_lmao": loss,
        }

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        input_features, selected_labels, indices = batch

        if isinstance(input_features, list):
            input_features = torch.nested.nested_tensor(input_features)

        if self.execute_once:
            for dataset_name, distribution in DATASET_DISTRIBUTIONS.items():
                DATASET_DISTRIBUTIONS[dataset_name] = distribution.to(
                    input_features.device
                )

            print("Number of datasets:", len(self.trainer.datamodule.dataset))
            self.execute_once = False

        if (
            dataloader_idx != self.current_dataloader_idx
            and len(self.trainer.datamodule.dataset) != 2
        ):
            self.log_and_reset_metrics()
            self.current_dataloader_idx = dataloader_idx

        self.update_prefix(
            (dataloader_idx + 1) % (len(self.trainer.datamodule.dataset) - 1)
        )

        loss, outputs = self(
            input_features,
            selected_labels,
            indices,
            train=False,
            dataloader_idx=dataloader_idx,
        )

        return {
            "valid_loss": loss,
        }

    def log_and_reset_metrics(self):
        self.log_validation()

        self.micro_accuracy.reset()
        self.weighted_accuracy.reset()
        self.micro_f1.reset()
        self.weighted_f1.reset()
        self.macro_f1.reset()
        self.precision_macro.reset()
        self.recall_macro.reset()

    def update_prefix(self, dataloader_idx):
        is_ood = dataloader_idx != 0
        dataset = dataset_name = self.trainer.datamodule.dataset[dataloader_idx + 1]
        dataset_name = dataset.dataset_name if is_ood else dataset.dataset.dataset_name

        self.prefix = f"{'OOD' if is_ood else 'ID'}_{dataset_name}"

    def on_train_batch_end(self, out, batch, batch_idx):
        self.log(
            f"train_loss",
            out["unscaled_lmao"],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=self.batch_size,
            sync_dist=True,
        )

    def on_validation_epoch_end(self):
        if len(self.trainer.datamodule.dataset) == 2:
            self.log_and_reset_metrics()
            return

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.trainer.model.model.parameters(),
            lr=1e-4,
            weight_decay=0.1,
        )

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=1e-4,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.1,
        )

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def log_validation(
        self,
    ):
        self.log(
            f"{self.prefix}_valid_accuracy",
            self.micro_accuracy,
            batch_size=self.batch_size,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            add_dataloader_idx=False,
        )
        self.log(
            f"{self.prefix}_valid_weighted_accuracy",
            self.weighted_accuracy,
            batch_size=self.batch_size,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            add_dataloader_idx=False,
        )

        self.log(
            f"{self.prefix}_valid_macro_f1",
            self.macro_f1,
            batch_size=self.batch_size,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            add_dataloader_idx=False,
        )
        self.log(
            f"{self.prefix}_valid_weighted_f1",
            self.weighted_f1,
            batch_size=self.batch_size,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            add_dataloader_idx=False,
        )
        self.log(
            f"{self.prefix}_valid_micro_f1",
            self.micro_f1,
            batch_size=self.batch_size,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            add_dataloader_idx=False,
        )

        self.log(
            f"{self.prefix}_valid_macro_precision",
            self.precision_macro,
            batch_size=self.batch_size,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            add_dataloader_idx=False,
        )
        self.log(
            f"{self.prefix}_valid_macro_recall",
            self.recall_macro,
            batch_size=self.batch_size,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            add_dataloader_idx=False,
        )

    def create_validation_metrics(self):
        self.micro_accuracy = torchmetrics.Accuracy(
            "multiclass",
            num_classes=8,
            average="micro",
        )
        self.weighted_accuracy = torchmetrics.Accuracy(
            "multiclass",
            num_classes=8,
            average="weighted",
        )
        self.micro_f1 = torchmetrics.F1Score(
            "multiclass",
            num_classes=8,
            average="micro",
        )
        self.weighted_f1 = torchmetrics.F1Score(
            "multiclass",
            num_classes=8,
            average="weighted",
        )
        self.macro_f1 = torchmetrics.F1Score(
            "multiclass",
            num_classes=8,
            average="macro",
        )
        self.precision_macro = torchmetrics.Precision(
            "multiclass",
            num_classes=8,
            average="macro",
        )
        self.recall_macro = torchmetrics.Recall(
            "multiclass",
            num_classes=8,
            average="macro",
        )

        self.class_f1 = torchmetrics.F1Score(
            "multiclass",
            num_classes=8,
            average=None,
        )

    def compute_validation_metrics(
        self, indices, features, pooled_output, reshaped_labels
    ):
        if self.current_dataloader_idx != 0:
            train_distribution = (
                self.trainer.datamodule.train_dataset.dataset.get_dataset_distribution()
            )

            valid_distribution = self.trainer.datamodule.dataset[
                self.current_dataloader_idx + 1
            ].get_dataset_distribution()

            pooled_output -= torch.log(train_distribution**1 + 1e-12)
            pooled_output += torch.log(valid_distribution**1 + 1e-12)

            other_class_idx = GLOBAL_LABEL_MAP["other"]
            pooled_output[:, other_class_idx] -= 1e12

            valid_indices = reshaped_labels[:, other_class_idx] == 0
            reshaped_labels = reshaped_labels[valid_indices]
            pooled_output = pooled_output[valid_indices]

            if len(reshaped_labels) == 0:
                return

        argmax_pooled_output = torch.argmax(pooled_output, dim=-1)
        argmax_labels = torch.argmax(reshaped_labels, dim=-1)

        self.micro_accuracy(argmax_pooled_output, argmax_labels)
        self.weighted_accuracy(argmax_pooled_output, argmax_labels)
        self.micro_f1(argmax_pooled_output, argmax_labels)
        self.weighted_f1(argmax_pooled_output, argmax_labels)
        self.macro_f1(argmax_pooled_output, argmax_labels)
        self.precision_macro(argmax_pooled_output, argmax_labels)
        self.recall_macro(argmax_pooled_output, argmax_labels)

        self.class_f1(argmax_pooled_output, argmax_labels)

    def apply_weight_decay(self):
        weight_decay = 0.01

        for param in self.trainer.model.model.head.parameters():
            if param.requires_grad and param.dim() > 1:
                param.data = param.data - weight_decay * param.data
