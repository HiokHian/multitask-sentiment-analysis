from multi_task_models import Model
from utils import (
    return_label_mappings,
    get_eval_dataset,
    get_class_weights,
    MultiTaskCrawledDataset,
    MultiTaskGoEmotionDataset,
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
import torch
from transformers import DistilBertTokenizer
from torch.utils.data import DataLoader
import pytorch_lightning as pl

tasks = ['polarity', 'subjectivity', 'emotion']
checkpoint_path = (
    "/data/ongh0068/nlp/polarity_subjectivity_emotion/2022-10-25_12_44_20.051793/epoch=03-averaged_f1_val=0.62.ckpt"
)
GPU_NUM = 1

from datasets import load_dataset

go_emotion_dataset = load_dataset("go_emotions")  # base dataset
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
for task in tasks:
    task_to_class_weights[task] = torch.FloatTensor(
        get_class_weights(*task_labels_mapping_dict[task], y_train)
    ).to(f"cuda:{GPU_NUM}")

metrics = {}
for task in tasks:
    metrics[task] = {}
    metrics[task]["F1Score"] = (
        MultilabelF1Score(num_labels=7).to(f"cuda:{GPU_NUM}")
        if task == "emotion"
        else BinaryF1Score().to(f"cuda:{GPU_NUM}")
        if task == "subjectivity"
        else MulticlassF1Score(num_classes=3).to(f"cuda:{GPU_NUM}")
    )
    metrics[task]["Recall"] = (
        MultilabelPrecision(num_labels=7).to(f"cuda:{GPU_NUM}")
        if task == "emotion"
        else BinaryPrecision().to(f"cuda:{GPU_NUM}")
        if task == "subjectivity"
        else MulticlassPrecision(num_classes=3).to(f"cuda:{GPU_NUM}")
    )
    metrics[task]["Precision"] = (
        MultilabelRecall(num_labels=7).to(f"cuda:{GPU_NUM}")
        if task == "emotion"
        else BinaryRecall().to(f"cuda:{GPU_NUM}")
        if task == "subjectivity"
        else MulticlassRecall(num_classes=3).to(f"cuda:{GPU_NUM}")
    )

model = Model.load_from_checkpoint(
    checkpoint_path, multiplexing = False, task_to_class_weights=task_to_class_weights, metrics = metrics, tasks=tasks
)
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

############### Try test set#########################
# from datasets import load_dataset

# go_emotion_dataset = load_dataset("go_emotions")  # base dataset
# X_test, y_test = (
#     go_emotion_dataset["train"]["text"],
#     go_emotion_dataset["train"]["labels"],
# )
# test_encodings = tokenizer(X_test, truncation=True, padding=True)
# test_dataset = MultiTaskGoEmotionDataset(test_encodings, y_test, task=task)
# test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=8)

############### Try test set#########################


X_test, y_test = get_eval_dataset()
test_encodings = tokenizer(X_test, truncation=True, padding=True)
test_dataset = MultiTaskCrawledDataset(test_encodings, y_test, tasks=tasks)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=8)
trainer = pl.Trainer(
    accelerator="gpu",
    devices=[GPU_NUM],
)
trainer.test(model, test_loader)
