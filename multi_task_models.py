from torchmetrics.classification import (
    BinaryF1Score,
    MultilabelF1Score,
    MulticlassF1Score,
    BinaryRecall,
    MultilabelRecall,
    MulticlassRecall,
    BinaryPrecision,
    MultilabelPrecision,
    MulticlassPrecision,
)
from multi_task_model_utils import BERTBackboneMultiTaskModel
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
import pytorch_lightning as pl
import torch

import random
import bisect

task_to_loss_mapping = {
    "polarity": CrossEntropyLoss,
    "subjectivity": BCEWithLogitsLoss,
    "emotion": BCEWithLogitsLoss,
}

# Define lightning module
class Model(pl.LightningModule):
    def __init__(
        self, tasks, multiplexing, task_to_class_weights, metrics, freeze=False
    ):
        super().__init__()
        self.tasks = tasks
        self.task_to_class_weights = task_to_class_weights
        self.metrics = metrics
        self.multiplexing = multiplexing
        self.model = BERTBackboneMultiTaskModel(tasks, multiplexing, freeze)
        self.losses = {}
        # polarity => put class weights in cross entropy loss
        # subjectivity => put pos weight = num_neg/num_pos in pos_weight = weight of pos/weight of neg
        # emotion => use BCEWithLogitsLoss reduction = none and multiply weights with loss
        for task in tasks:
            print(f"using class_weights for {task} {self.task_to_class_weights[task]}")
            if task == "polarity":
                self.losses[task] = task_to_loss_mapping[task](
                    weight=self.task_to_class_weights[task]
                )
            elif task == "subjectivity":
                self.losses[task] = task_to_loss_mapping[task](
                    pos_weight=self.task_to_class_weights[task][1]
                    / self.task_to_class_weights[task][0]
                )
            else:
                self.losses[task] = task_to_loss_mapping[task](reduction="none")

        self.idx_to_class_mapping = {
            i: task for task, i in zip(self.tasks, range(0, len(self.tasks)))
        }  # eg {task_1: 1, task_2: 2, task_3: 3}
        self.random_intervals = [
            i * 1 / len(self.tasks)
            for i in range(1, len(self.idx_to_class_mapping) + 1)
        ]  # generate random intervals for each task; for 3 datasets, this will be [0.33, 0.66, 1.0]


    def forward(self, input_ids, attention_mask, tasks):
        return self.model(input_ids, attention_mask, tasks)

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx, split="train")
        return loss

    def _step(self, batch, batch_idx, split):
        if split == 'train' and self.multiplexing:
            # draw random number and pick a single task
            random_sample = random.random()
            idx = bisect.bisect_left(self.random_intervals, random_sample)
            tasks = [self.idx_to_class_mapping[idx]]
        else: # if validation or test or not multiplexing
            tasks = self.tasks

        # unpack all tasks if not multiplexing, otherwise, unpack only that one task
        input_ids, attention_mask, labels = (
            batch["input_ids"],
            batch["attention_mask"],
            {task: batch[f"{task}_labels"].float() for task in tasks},
        )

        preds = self.model(
            input_ids, attention_mask, [task for task in tasks], split = split
        )  # output will either be dictionary (not multiplexing) or just one output (multiplexing)
        losses = {}
        metrics = {}
        for task in tasks:
            if task == "polarity":
                losses[task] = self.losses[task](
                    preds[task], torch.max(labels[task], 1)[1]
                )
            elif task == "subjectivity":
                losses[task] = self.losses[task](preds[task], labels[task])
            elif task == "emotion":
                losses[task] = (
                    self.losses[task](preds[task], labels[task])
                    * self.task_to_class_weights[task]
                ).mean()

            f1, recall, precision = self._log_metrics(
                preds[task], labels[task], split, task
            )
            metrics[task] = {}
            metrics[task]["f1"] = f1
            metrics[task]["recall"] = recall
            metrics[task]["precision"] = precision
        loss = sum(losses.values())
        self.log(f"{split}_averaged_loss", loss)
        averaged_metrics = {}
        for metric_name in metrics[tasks[0]].keys():
            averaged_metrics[metric_name] = []
            for task in tasks:
                averaged_metrics[metric_name].append(metrics[task][metric_name])
            self.log(
                f"averaged_{metric_name}_{split}",
                sum(averaged_metrics[metric_name]) / len(tasks),
                on_epoch=True,
                prog_bar=True,
            )

        return loss

    def _log_metrics(self, preds, labels, split, task):
        if task == "polarity":
            preds_ind = torch.max(preds, 1)[1].to(preds.device)
            labels_ind = torch.max(labels, 1)[1].cuda(labels.device)

            f1 = self.metrics[task]["F1Score"](preds_ind, labels_ind)
            recall = self.metrics[task]["Recall"](preds_ind, labels_ind)
            precision = self.metrics[task]["Precision"](preds_ind, labels_ind)
        else:
            f1 = self.metrics[task]["F1Score"](preds, labels)
            recall = self.metrics[task]["Recall"](preds, labels)
            precision = self.metrics[task]["Precision"](preds, labels)

        self.log(f"{task}_{split}_f1", f1, on_epoch=True, prog_bar=False)
        self.log(f"{task}_{split}_recall", recall, on_epoch=True, prog_bar=False)
        self.log(f"{task}_{split}_precision", precision, on_step=True, prog_bar=False)
        return f1, recall, precision

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx, split="val")

    def test_step(self, batch, batch_idx):
        self._step(batch, batch_idx, split="test")

    def configure_optimizers(self):
        # TODO: include differential learning rate https://github.com/Lightning-AI/lightning/issues/2005
        lrs = [5e-6, 5e-4]
        betas = [0.9, 0.999]
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": self.model.bert_backbone.parameters(),
                    "lr": lrs[0],
                    "name": "bert_backbone",
                }
            ]
            + [
                {
                    "params": self.model.classifiers[task].parameters(),
                    "lr": lrs[1],
                    "name": f"{task}_classifier",
                }
                for task in self.model.classifiers.keys()
            ],
            lr=5e-5,
            betas=(betas[0], betas[1]),
        )

        base_lrs = [lrs[0]] + [
            lrs[1] for _ in range(len(self.model.classifiers.keys()))
        ]
        lr_scheduler = {
            "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=[lr * 10.0 for lr in base_lrs],
                total_steps=self.trainer.max_epochs
                * len(
                    self.trainer._data_connector._train_dataloader_source.dataloader()
                ),
                div_factor=25.0,
                three_phase=True,
            ),
            "name": "scheduler_lr",
            "interval": "step",
            "frequency": self.trainer.max_epochs,
        }
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
