from datasets import load_dataset
from utils import (
    return_label_mappings,
    reverse_mapping,
    indices_to_one_hot,
    get_class_weights,
    get_eval_dataset,
    MultiTaskGoEmotionDataset,
    MultiTaskCrawledDataset,
)
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
from transformers import DistilBertTokenizer
import torch
from datasets import load_dataset
from multi_task_models import Model

go_emotion_dataset = load_dataset("go_emotions")  # base dataset

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

X_train, X_valid, X_test = (
    go_emotion_dataset["train"]["text"][:],
    go_emotion_dataset["validation"]["text"][:],
    go_emotion_dataset["test"]["text"],
)
y_train, y_valid, y_test = (
    go_emotion_dataset["train"]["labels"][:],
    go_emotion_dataset["validation"]["labels"][:],
    go_emotion_dataset["test"]["labels"],
)

# concat train and val and treat test as val ####################
X_train = X_train + X_valid
y_train = y_train + y_valid
# X_valid, y_valid = X_test, y_valid
X_valid, y_valid = get_eval_dataset()
# concat train and val and treat test as val ####################

train_encodings = tokenizer(X_train, truncation=True, padding=True)
valid_encodings = tokenizer(X_valid, truncation=True, padding=True)

ALLOWED_TASKS = ["polarity", "subjectivity", "emotion"]


# define model and lightning module

import pytorch_lightning as pl
from torch.utils.data import DataLoader


import sys

# EXPECTED COMMAND: python multitask.py polarity,subjectivity 16 False
TASKS = sys.argv[1].split(",")  # change this
BATCH_SIZE = int(sys.argv[2])
MULTIPLEXING = bool(sys.argv[2])

print(
    f"RUNNING: {TASKS} TASKS with BATCH SIZE {BATCH_SIZE}; Using Multiplexing: {MULTIPLEXING}"
)


train_dataset = MultiTaskGoEmotionDataset(train_encodings, y_train, tasks=TASKS)
val_dataset = MultiTaskGoEmotionDataset(valid_encodings, y_valid, tasks=TASKS)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8
)
valid_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8
)

(
    ekman_mapping,
    ekman_labels,
    polarity_mapping,
    polarity_labels,
    subjectivity_mapping,
    subjectivity_labels,
    LABELS,
) = return_label_mappings()

task_labels_mapping_dict = {
    "polarity": [polarity_mapping, polarity_labels],
    "subjectivity": [subjectivity_mapping, subjectivity_labels],
    "emotion": [ekman_mapping, ekman_labels],
}


task_to_class_weights = {}
for task in TASKS:
    task_to_class_weights[task] = torch.FloatTensor(
        get_class_weights(*task_labels_mapping_dict[task], y_train)
    ).to("cuda:1")

metrics = {}
for task in TASKS:
    metrics[task] = {}
    metrics[task]["F1Score"] = (
        MultilabelF1Score(num_labels=7).to("cuda:1")
        if task == "emotion"
        else BinaryF1Score().to("cuda:1")
        if task == "subjectivity"
        else MulticlassF1Score(num_classes=3).to("cuda:1")
    )
    metrics[task]["Recall"] = (
        MultilabelPrecision(num_labels=7).to("cuda:1")
        if task == "emotion"
        else BinaryPrecision().to("cuda:1")
        if task == "subjectivity"
        else MulticlassPrecision(num_classes=3).to("cuda:1")
    )
    metrics[task]["Precision"] = (
        MultilabelRecall(num_labels=7).to("cuda:1")
        if task == "emotion"
        else BinaryRecall().to("cuda:1")
        if task == "subjectivity"
        else MulticlassRecall(num_classes=3).to("cuda:1")
    )

model = Model(
    tasks=TASKS,
    task_to_class_weights=task_to_class_weights,
    metrics=metrics,
    multiplexing=MULTIPLEXING,
    freeze=False,
)  # change this

from datetime import datetime
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor

now = str(datetime.now()).replace(" ", "_").replace(":", "_")
print(now)


lr_monitor = LearningRateMonitor(logging_interval="step")
tensorboard_logger = TensorBoardLogger(
    save_dir=f'{"_".join(TASKS)}/{now}', name=f"logs_{now}_{BATCH_SIZE}"
)
early_stopping = EarlyStopping(monitor="val_averaged_loss", patience=10)
metric_to_track = f"{TASKS[0]}_averaged_f1_val"
checkpoint_callback = ModelCheckpoint(
    save_top_k=1,
    monitor="val_averaged_loss",
    dirpath=f'{"_".join(TASKS)}/{now}',
    mode="min",
    filename="{epoch:02d}-{" + metric_to_track + ":.2f}",
)


# TODO: add model checkpoint callback and use it to monitor F1 score (needs to be done after adding evaluation metrics); run a few times to ensure model is being saved in the directory https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.callbacks.ModelCheckpoint.html#pytorch_lightning.callbacks.ModelCheckpoint
trainer = pl.Trainer(
    accelerator="gpu",
    devices=[1],
    callbacks=[checkpoint_callback, lr_monitor, early_stopping],
    logger=tensorboard_logger,
    max_epochs=20,
)


trainer.fit(model, train_loader, valid_loader)
