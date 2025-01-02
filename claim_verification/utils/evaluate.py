import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import torch


def evaluate(outputs, targets):
    confusion = torch.zeros(3,3, dtype=torch.long)
    for i in range(len(outputs)):
        confusion[outputs[i], targets[i]] += 1
    return {
        'macro_f1': f1_score(targets, outputs, zero_division=0, average='macro'),
        'f1': tuple(f1_score(targets, outputs, zero_division=0, average=None)),
        'precision': tuple(precision_score(targets, outputs, zero_division=0, average=None)),
        'recall': tuple(recall_score(targets, outputs, zero_division=0, average=None)),
        'accuracy': accuracy_score(targets, outputs),
        'confusion': confusion
    }

def oracle_evaluate(outputs1, outputs2, targets):
    correct = 0
    for i in range(len(outputs1)):
        if outputs1[i] == targets[i] or outputs2[i] == targets[i]:
            correct += 1
    return correct / len(outputs1)



