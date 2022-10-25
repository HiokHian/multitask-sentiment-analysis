from models import Model
from utils import (
    get_eval_dataset,
    SingleTaskCrawledDataset,
    SingleTaskGoEmotionDataset,
)
import torch
from transformers import DistilBertTokenizer
from torch.utils.data import DataLoader
import pytorch_lightning as pl

task = "emotion"
checkpoint_path = (
    "/data/ongh0068/nlp/emotion/2022-10-21_14_21_23.327200/epoch=02-val_f1=0.37.ckpt"
)


class_weights = None
model = Model.load_from_checkpoint(
    checkpoint_path, class_weights=class_weights, task=task
)
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

############### Try test set#########################
from datasets import load_dataset

go_emotion_dataset = load_dataset("go_emotions")  # base dataset
X_test, y_test = (
    go_emotion_dataset["train"]["text"],
    go_emotion_dataset["train"]["labels"],
)
test_encodings = tokenizer(X_test, truncation=True, padding=True)
test_dataset = SingleTaskGoEmotionDataset(test_encodings, y_test, task=task)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=8)

############### Try test set#########################


# X_test, y_test = get_eval_dataset()
# test_encodings = tokenizer(X_test, truncation=True, padding=True)
# test_dataset = SingleTaskCrawledDataset(test_encodings, y_test, task=task)
# test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=8)
trainer = pl.Trainer(
    accelerator="gpu",
    devices=[2],
)
trainer.test(model, test_loader)
