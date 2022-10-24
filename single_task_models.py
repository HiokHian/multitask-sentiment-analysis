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

from single_task_model_utils import BERTBackboneSingleTaskModel
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
import pytorch_lightning as pl
import torch


# Define lightning module
class Model(pl.LightningModule):
    def __init__(self, task, class_weights, freeze=False):
        super().__init__()
        self.model = BERTBackboneSingleTaskModel(task=task, freeze=freeze)
        self.class_weights = class_weights
        print("using class_weights ", self.class_weights)
        # polarity => put class weights in cross entropy loss
        # subjectivity => put pos weight = num_neg/num_pos in pos_weight = weight of pos/weight of neg
        # emotion => use BCEWithLogitsLoss reduction = none and multiply weights with loss

        if task == "polarity":
            self.loss = CrossEntropyLoss(weight=self.class_weights)
        elif task == "subjectivity":
            self.loss = BCEWithLogitsLoss(
                pos_weight=self.class_weights[1] / self.class_weights[0]
            )
        else:
            self.loss = BCEWithLogitsLoss(reduction="none")

        self.F1Score = (
            MultilabelF1Score(num_labels=7)
            if task == "emotion"
            else BinaryF1Score()
            if task == "subjectivity"
            else MulticlassF1Score(num_classes=3)
        )
        self.Recall = (
            MultilabelPrecision(num_labels=7)
            if task == "emotion"
            else BinaryPrecision()
            if task == "subjectivity"
            else MulticlassPrecision(num_classes=3)
        )
        self.Precision = (
            MultilabelRecall(num_labels=7)
            if task == "emotion"
            else BinaryRecall()
            if task == "subjectivity"
            else MulticlassRecall(num_classes=3)
        )
        self.task = task

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask)

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx, split="train")
        return loss

    def _step(self, batch, batch_idx, split):
        input_ids, attention_mask, labels = (
            batch["input_ids"],
            batch["attention_mask"],
            batch["labels"].float(),
        )
        preds = self.model(input_ids, attention_mask=attention_mask)

        self._log_metrics(preds, labels, split)
        if split != "test":
            if self.task == "polarity":
                loss = self.loss(preds, torch.max(labels, 1)[1])
            elif self.task == "subjectivity":
                loss = self.loss(preds, labels)
            elif self.task == "emotion":
                loss = (self.loss(preds, labels) * self.class_weights).mean()

            self.log(f"{split}_loss", loss)

            return loss

    def _log_metrics(self, preds, labels, split):
        if self.task == "polarity":
            preds_ind = torch.max(preds, 1)[1]
            labels_ind = torch.max(labels, 1)[1]

            f1 = self.F1Score(preds_ind, labels_ind)
            recall = self.Recall(preds_ind, labels_ind)
            precision = self.Precision(preds_ind, labels_ind)
        else:
            f1 = self.F1Score(preds, labels)
            recall = self.Recall(preds, labels)
            precision = self.Precision(preds, labels)

        self.log(f"{split}_f1", f1, on_epoch=True, prog_bar=True)
        self.log(f"{split}_recall", recall, on_epoch=True, prog_bar=True)
        self.log(f"{split}_precision", precision, on_step=True, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx, split="val")

    def test_step(self, batch, batch_idx):
        self._step(batch, batch_idx, split="test")

    def configure_optimizers(self):
        # TODO: include differential learning rate https://github.com/vdouet/Discriminative-learning-rates-PyTorch
        # TODO: change Adam to OneCycle
        lrs = [5e-6, 5e-4]
        betas = [0.9, 0.999]
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": self.model.bert_backbone.parameters(),
                    "lr": lrs[0],
                    "name": "bert_backbone",
                },
                {
                    "params": self.model.classifier.parameters(),
                    "lr": lrs[1],
                    "name": "classifier",
                },
            ],
            lr=5e-5,
            betas=(betas[0], betas[1]),
        )
        print("max_epochs", self.trainer.max_epochs)

        lr_scheduler = {
            "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=[lr * 10.0 for lr in lrs],
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
