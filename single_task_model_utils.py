import torch
from transformers import DistilBertModel, RobertaModel, BertModel

tasks_to_n_labels = {"polarity": 3, "subjectivity": 1, "emotion": 7}


class MLPClassifier(torch.nn.Module):
    def __init__(self, task, num_hidden_layers=2):
        super(MLPClassifier, self).__init__()
        self.linear_layer = torch.nn.Linear(768, 256)
        self.hidden_layers = torch.nn.ModuleList(
            [torch.nn.Linear(256, 256) for _ in range(num_hidden_layers)]
        )
        self.dropout_layers = torch.nn.ModuleList(
            [torch.nn.Dropout(0.4) for _ in range(num_hidden_layers)]
        )
        self.final_dropout_layer = torch.nn.Dropout(0.5)
        self.n_labels = 7 if task == "emotion" else 1 if task == "subjectivity" else 3
        self.final_layer = torch.nn.Linear(256, tasks_to_n_labels[task])

    def forward(self, x):
        x = self.linear_layer(x)
        for dropout, layer in zip(self.hidden_layers, self.dropout_layers):
            x = dropout(x)
            x = layer(x)
        x = self.final_dropout_layer(x)
        x = self.final_layer(x)

        return x


class BERTBackboneSingleTaskModel(torch.nn.Module):
    def __init__(self, task, freeze=False):
        super(BERTBackboneSingleTaskModel, self).__init__()

        # TRY OTHER MODELS ####################
        # self.bert_backbone  = BertModel.from_pretrained("bert-base-uncased")
        # TRY OTHER MODELS ####################

        self.bert_backbone = DistilBertModel.from_pretrained(
            "distilbert-base-uncased"
        )  # , output_attentions=True)
        if freeze:
            for param in self.bert_backbone.parameters():
                param.requires_grad = False
        self.classifier = MLPClassifier(task=task)

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert_backbone(
            input_ids=input_ids, attention_mask=attention_mask
        )
        hidden_state = bert_output[0]
        x = hidden_state[:, 0]
        return self.classifier(x)
