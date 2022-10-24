# https://github.com/google-research/google-research/tree/master/goemotions/data
import pandas as pd
from collections import defaultdict, Counter, OrderedDict
import torch

# NOTE: neutral has been added to all the mappings
# NOTE: ambiguous is mapped to 0 for polarity and to 0 for subjectivity
ekman_mapping = {
    "anger": ["anger", "annoyance", "disapproval"],
    "disgust": ["disgust"],
    "fear": ["fear", "nervousness"],
    "joy": [
        "joy",
        "amusement",
        "approval",
        "excitement",
        "gratitude",
        "love",
        "optimism",
        "relief",
        "pride",
        "admiration",
        "desire",
        "caring",
    ],
    "sadness": ["sadness", "disappointment", "embarrassment", "grief", "remorse"],
    "surprise": ["surprise", "realization", "confusion", "curiosity"],
    "neutral": ["neutral"],
}

# to be one hot encoded
ekman_labels = {
    "anger": 0,
    "disgust": 1,
    "fear": 2,
    "joy": 3,
    "sadness": 4,
    "surprise": 5,
    "neutral": 6,
}

polarity_mapping = {
    "positive": [
        "amusement",
        "excitement",
        "joy",
        "love",
        "desire",
        "optimism",
        "caring",
        "pride",
        "admiration",
        "gratitude",
        "relief",
        "approval",
    ],
    "negative": [
        "fear",
        "nervousness",
        "remorse",
        "embarrassment",
        "disappointment",
        "sadness",
        "grief",
        "disgust",
        "anger",
        "annoyance",
        "disapproval",
    ],
    "ambiguous": ["realization", "surprise", "curiosity", "confusion", "neutral"],
}

polarity_labels = {
    "positive": 1,
    "negative": 2,
    "ambiguous": 0,
}

subjectivity_mapping = {
    "subjective": [
        "amusement",
        "excitement",
        "joy",
        "love",
        "desire",
        "optimism",
        "caring",
        "pride",
        "admiration",
        "gratitude",
        "relief",
        "approval",
        "fear",
        "nervousness",
        "remorse",
        "embarrassment",
        "disappointment",
        "sadness",
        "grief",
        "disgust",
        "anger",
        "annoyance",
        "disapproval",
    ],
    "neutral": ["realization", "surprise", "curiosity", "confusion", "neutral"],
}

subjectivity_labels = {
    "subjective": 1,
    "neutral": 0,
}

LABELS = [
    "admiration",
    "amusement",
    "anger",
    "annoyance",
    "approval",
    "caring",
    "confusion",
    "curiosity",
    "desire",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "excitement",
    "fear",
    "gratitude",
    "grief",
    "joy",
    "love",
    "nervousness",
    "optimism",
    "pride",
    "realization",
    "relief",
    "remorse",
    "sadness",
    "surprise",
    "neutral",
]


def return_label_mappings():
    return (
        ekman_mapping,
        ekman_labels,
        polarity_mapping,
        polarity_labels,
        subjectivity_mapping,
        subjectivity_labels,
        LABELS,
    )


def reverse_mapping(mapping, task_labels):
    return {
        LABELS.index(v): task_labels[k]
        for k, list_values in mapping.items()
        for v in list_values
    }


# https://stackoverflow.com/questions/37292872/how-can-i-one-hot-encode-in-python
import numpy as np
from sklearn.utils.class_weight import compute_class_weight


def indices_to_one_hot(data, nb_classes):
    """Convert an iterable of indices to one-hot encoded labels."""
    targets = np.array(data).reshape(-1)
    return np.sum(np.eye(nb_classes)[targets], axis=0).astype(bool).astype(int)


def get_class_weights(task_mapping, task_labels, y_train):
    label_mapping = reverse_mapping(task_mapping, task_labels)
    label_distribution = defaultdict(int)
    for labels in y_train:
        for tmp in labels:
            label_distribution[label_mapping[tmp]] += 1
    label_distribution = OrderedDict(sorted(label_distribution.items()))
    y = [[k] * v for k, v in label_distribution.items()]
    y = [item for sublist in y for item in sublist]
    weights = compute_class_weight(
        "balanced", classes=list(label_distribution.keys()), y=y
    )
    return weights


import math


# Dataset
def get_eval_dataset(eval_file_path="sentences_with_suggestion_v1.csv"):
    df_eval = pd.read_csv(eval_file_path).replace("netural", "neutral")

    # # remove all autolabelled samples
    # df_eval = df_eval.dropna()
    # # remove all autolabelled samples

    labels, suggested_labels = (
        df_eval["label"].tolist(),
        df_eval["suggested_label"].tolist(),
    )

    for i in range(len(labels)):
        if type(labels[i]) == str:
            pass
        elif math.isnan(labels[i]):
            labels[i] = str(ekman_labels[suggested_labels[i]])
    preprocessed_labels = []
    for label in labels:
        if "," in label:
            splitted = [int(i) for i in label.split(",")]
            preprocessed_labels += [splitted]
        else:
            preprocessed_labels += [[int(label)]]

    return df_eval["text"].tolist(), preprocessed_labels


from torch.utils.data import Dataset

ALLOWED_TASKS = ["polarity", "subjectivity", "emotion"]

# https://huggingface.co/transformers/v3.2.0/custom_datasets.html
class SingleTaskGoEmotionDataset(Dataset):
    def __init__(self, encodings, labels, task):
        self.encodings = encodings
        self.labels = labels
        assert task in ALLOWED_TASKS
        self.task = task
        self.orig_emotion_to_polarity_mapping = reverse_mapping(
            polarity_mapping, polarity_labels
        )
        self.orig_emotion_to_subjectivity_mapping = reverse_mapping(
            subjectivity_mapping, subjectivity_labels
        )
        self.orig_emotion_to_ekman_emotion_mapping = reverse_mapping(
            ekman_mapping, ekman_labels
        )
        self.task_mapping = {
            "polarity": self.orig_emotion_to_polarity_mapping,
            "subjectivity": self.orig_emotion_to_subjectivity_mapping,
            "emotion": self.orig_emotion_to_ekman_emotion_mapping,
        }

    def __getitem__(self, idx):
        item = {key: torch.tensor(value[idx]) for key, value in self.encodings.items()}
        task = self.task
        if task == "emotion":
            mapped_labels = [self.task_mapping[task][i] for i in self.labels[idx]]
            item["labels"] = torch.tensor(
                indices_to_one_hot(mapped_labels, len(ekman_labels))
            )
        elif task == "polarity":
            mapped_labels = [self.task_mapping[task][i] for i in self.labels[idx]]
            item["labels"] = torch.tensor(
                indices_to_one_hot(mapped_labels, len(polarity_labels))
            )
        else:  # normal mapping no need to one hot
            item["labels"] = torch.tensor(
                [self.task_mapping[task][self.labels[idx][0]]]
            )
        return item

    def __len__(self):
        return len(self.labels)


ekman_emotion_to_polarity_mapping = {
    0: 2,
    1: 2,
    2: 2,
    3: 1,
    4: 2,
    5: 0,
    6: 0,
}
ekman_emotion_to_subjectivity_mapping = {
    0: 1,
    1: 1,
    2: 1,
    3: 1,
    4: 1,
    5: 0,
    6: 0,
}

# subjectivity_labels = {
#     "subjective": 1,
#     "neutral": 0,
# }
# polarity_labels = {
#     "positive": 1,
#     "negative": 2,
#     "ambiguous": 0,
# }
# ekman_labels = {
#     "anger": 0,
#     "disgust": 1,
#     "fear": 2,
#     "joy": 3,
#     "sadness": 4,
#     "surprise": 5, # neutral
#     "neutral": 6, # neutral
# }


class SingleTaskCrawledDataset(Dataset):
    def __init__(self, encodings, labels, task):
        self.encodings = encodings
        self.labels = labels
        assert task in ALLOWED_TASKS
        self.task = task
        self.ekman_emotion_to_polarity_mapping = ekman_emotion_to_polarity_mapping
        self.ekman_emotion_to_subjectivity_mapping = (
            ekman_emotion_to_subjectivity_mapping
        )
        self.task_mapping = {
            "polarity": self.ekman_emotion_to_polarity_mapping,
            "subjectivity": self.ekman_emotion_to_subjectivity_mapping,
        }

    def __getitem__(self, idx):
        item = {key: torch.tensor(value[idx]) for key, value in self.encodings.items()}
        task = self.task
        if task == "emotion":  # no need to do anything
            mapped_labels = self.labels[idx]
            item["labels"] = torch.tensor(
                indices_to_one_hot(mapped_labels, len(ekman_labels))
            )
        elif task == "polarity":  #
            mapped_labels = [self.task_mapping[task][i] for i in self.labels[idx]]
            item["labels"] = torch.tensor(
                indices_to_one_hot(mapped_labels, len(polarity_labels))
            )
        else:  # normal mapping no need to one hot
            item["labels"] = torch.tensor(
                [self.task_mapping[task][self.labels[idx][0]]]
            )
        return item

    def __len__(self):
        return len(self.labels)


# https://huggingface.co/transformers/v3.2.0/custom_datasets.html
class MultiTaskGoEmotionDataset(Dataset):
    def __init__(self, encodings, labels, tasks):
        self.encodings = encodings
        self.labels = labels
        assert all(task in ALLOWED_TASKS for task in tasks)
        self.tasks = tasks
        self.orig_emotion_to_polarity_mapping = reverse_mapping(
            polarity_mapping, polarity_labels
        )
        self.orig_emotion_to_subjectivity_mapping = reverse_mapping(
            subjectivity_mapping, subjectivity_labels
        )
        self.orig_emotion_to_ekman_emotion_mapping = reverse_mapping(
            ekman_mapping, ekman_labels
        )
        self.task_mapping = {
            "polarity": self.orig_emotion_to_polarity_mapping,
            "subjectivity": self.orig_emotion_to_subjectivity_mapping,
            "emotion": self.orig_emotion_to_ekman_emotion_mapping,
        }

    def __getitem__(self, idx):
        item = {key: torch.tensor(value[idx]) for key, value in self.encodings.items()}
        for task in self.tasks:
            if task == "emotion":
                mapped_labels = [self.task_mapping[task][i] for i in self.labels[idx]]
                item[f"{task}_labels"] = torch.tensor(
                    indices_to_one_hot(mapped_labels, len(ekman_labels))
                )
            elif task == "polarity":
                mapped_labels = [self.task_mapping[task][i] for i in self.labels[idx]]
                item[f"{task}_labels"] = torch.tensor(
                    indices_to_one_hot(mapped_labels, len(polarity_labels))
                )
            else:  # normal mapping no need to one hot
                item[f"{task}_labels"] = torch.tensor(
                    [self.task_mapping[task][self.labels[idx][0]]]
                )

        return item

    def __len__(self):
        return len(self.labels)


class MultiTaskCrawledDataset(Dataset):
    def __init__(self, encodings, labels, tasks):
        self.encodings = encodings
        self.labels = labels
        assert all(task in ALLOWED_TASKS for task in tasks)
        self.tasks = tasks
        self.ekman_emotion_to_polarity_mapping = ekman_emotion_to_polarity_mapping
        self.ekman_emotion_to_subjectivity_mapping = (
            ekman_emotion_to_subjectivity_mapping
        )
        self.task_mapping = {
            "polarity": self.ekman_emotion_to_polarity_mapping,
            "subjectivity": self.ekman_emotion_to_subjectivity_mapping,
        }

    def __getitem__(self, idx):
        item = {key: torch.tensor(value[idx]) for key, value in self.encodings.items()}
        for task in self.tasks:
            if task == "emotion":  # do nothing since it is already in ekman mapping
                mapped_labels = self.labels[idx]
                item[f"{task}_labels"] = torch.tensor(
                    indices_to_one_hot(mapped_labels, len(ekman_labels))
                )
            elif task == "polarity":
                mapped_labels = [self.task_mapping[task][i] for i in self.labels[idx]]
                item[f"{task}_labels"] = torch.tensor(
                    indices_to_one_hot(mapped_labels, len(polarity_labels))
                )
            else:  # normal mapping no need to one hot
                item[f"{task}_labels"] = torch.tensor(
                    [self.task_mapping[task][self.labels[idx][0]]]
                )

        return item

    def __len__(self):
        return len(self.labels)
