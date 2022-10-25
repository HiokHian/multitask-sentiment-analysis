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
        self.final_layer = torch.nn.Linear(256, tasks_to_n_labels[task])
        # for polarity or subjectivity, this will squash that 1 output to between 0 and 1; for go emotions, where it can have multiple emotions, this will squash all 5 output neurons between 1 and 0

    def forward(self, x):
        x = self.linear_layer(x)
        for dropout, layer in zip(self.hidden_layers, self.dropout_layers):
            x = dropout(x)
            x = layer(x)
        x = self.final_dropout_layer(x)
        x = self.final_layer(x)
        return x


class BERTBackboneMultiTaskModel(torch.nn.Module):
    def __init__(self, tasks, multiplexing=False, freeze=False):
        super(BERTBackboneMultiTaskModel, self).__init__()
        self.multiplexing = multiplexing
        self.bert_backbone = DistilBertModel.from_pretrained("distilbert-base-uncased")
        if freeze:
            for param in self.bert_backbone.parameters():
                param.requires_grad = False
        self.classifiers = torch.nn.ModuleDict(
            [[task, MLPClassifier(task)] for task in tasks]
        )

    def forward(self, input_ids, attention_mask, tasks, split):
        bert_output = self.bert_backbone(
            input_ids=input_ids, attention_mask=attention_mask
        )
        hidden_state = bert_output[0]
        x = hidden_state[:, 0]
        if split == 'train' and self.multiplexing:  # if multiplexing then only pass through relevant MLP  
            task = tasks[0]
            return {task: self.classifiers[task](x)}
        else: # if not multiplexing, pass through all MLPs
            return {task: self.classifiers[task](x) for task in tasks}
