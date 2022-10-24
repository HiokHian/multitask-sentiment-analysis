from utils import (
    return_label_mappings,
    reverse_mapping,
    indices_to_one_hot,
    get_class_weights,
    get_eval_dataset,
    SingleTaskGoEmotionDataset,
    SingleTaskCrawledDataset,
)

from transformers import DistilBertTokenizer, RobertaTokenizer, BertTokenizer
import torch
from datasets import load_dataset
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from single_task_models import Model

import sys
from datetime import datetime
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor

go_emotion_dataset = load_dataset("go_emotions")  # base dataset

# TRY OTHER MODELS ####################
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
# TRY OTHER MODELS ####################

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


ALLOWED_TASKS = ["polarity", "subjectivity", "emotion"]

train_encodings = tokenizer(X_train, truncation=True, padding=True)
valid_encodings = tokenizer(X_valid, truncation=True, padding=True)


# define model and lightning module
TASK = sys.argv[1]  # change this
BATCH_SIZE = int(sys.argv[2])

print(f"RUNNING: {TASK} TASK with BATCH SIZE {BATCH_SIZE}")
train_dataset = SingleTaskGoEmotionDataset(train_encodings, y_train, task=TASK)
val_dataset = SingleTaskCrawledDataset(
    valid_encodings, y_valid, task=TASK
)  # CHANGE THIS
# test_dataset = SingleTaskGoEmotionDataset(test_encodings, y_test, task=TASK)


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

class_weights = torch.FloatTensor(
    get_class_weights(*task_labels_mapping_dict[TASK], y_train)
).to("cuda:2")

model = Model(task=TASK, class_weights=class_weights, freeze=False)  # change this


now = str(datetime.now()).replace(" ", "_").replace(":", "_")
print(now)

lr_monitor = LearningRateMonitor(logging_interval="step")
tensorboard_logger = TensorBoardLogger(
    save_dir=f"{TASK}/{now}", name=f"logs_{now}_{BATCH_SIZE}"
)
early_stopping = EarlyStopping(monitor="val_loss", patience=10)
checkpoint_callback = ModelCheckpoint(
    save_top_k=1,
    monitor="val_loss",
    dirpath=f"{TASK}/{now}",
    mode="min",
    filename="{epoch:02d}-{val_f1:.2f}",
)


# TODO: add model checkpoint callback and use it to monitor F1 score (needs to be done after adding evaluation metrics); run a few times to ensure model is being saved in the directory https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.callbacks.ModelCheckpoint.html#pytorch_lightning.callbacks.ModelCheckpoint
trainer = pl.Trainer(
    accelerator="gpu",
    devices=[2],
    callbacks=[checkpoint_callback, lr_monitor, early_stopping],
    logger=tensorboard_logger,
    max_epochs=20,
)

trainer.fit(model, train_loader, valid_loader)
