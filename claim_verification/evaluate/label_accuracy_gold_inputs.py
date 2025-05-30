import argparse
import jsonlines

from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, accuracy_score

from claim_verification.utils.Functions import printf

parser = argparse.ArgumentParser()
parser.add_argument('--corpus', type=str, required=True)
parser.add_argument('--datasets', type=str, required=True)
parser.add_argument('--label-prediction', type=str, required=True)
parser.add_argument('--filter', type=str, choices=['structured', 'unstructured'])
args = parser.parse_args()

corpus = {doc['doc_id']: doc for doc in jsonlines.open(args.corpus)}
dataset = jsonlines.open(args.dataset)
label_prediction = jsonlines.open(args.label_prediction)


LABELS = {'CONTRADICT': 0, 'NOT_ENOUGH_INFO': 1, 'SUPPORT': 2}

n_total = 0
n_correct = 0

total = 0
corr = 0

targets = []
outputs1 = []
outputs2 =[]

def judege(pred_labels, weights):
    result = {"NOT_ENOUGH_INFO":0, "SUPPORT":0, "CONTRADICT":0}
    for i in range(len(pred_labels)):
        result[pred_labels[i]] += weights[i]
    sorted_result = sorted(result.items(), key=lambda x: x[1], reverse=True)

    return sorted_result[0][0]

abcd = 0
efig = 0
for data, prediction in zip(dataset, label_prediction):
    assert data['id'] == prediction['claim_id']

    if not data["evidence"]:
        gold_label = "NOT_ENOUGH_INFO"
    else:
        gold_labels = set()
        for entry in data["evidence"].values():
            for rat in entry:
                gold_labels.add(rat["label"])
        assert len(gold_labels) == 1
        gold_label = next(iter(gold_labels))

    pred_labels = [x["label"] for x in prediction["labels"].values()]
    confidences = [x["confidence"] for x in prediction["labels"].values()]
    pred = judege(pred_labels,confidences)
    pred2 =judege(pred_labels, [1]*len(pred_labels))
    n_total += len(pred_labels)
    total += 1
    correct = [x for x in pred_labels if x == gold_label]
    n_correct += len(correct)
    if len(correct) > 0:
        corr += 1
    # if pred == gold_label:
    #     abcd += 1
    # if pred2 == gold_label:
    #     efig += 1
    targets.append(gold_label)
    if pred == gold_label and pred2 != gold_label:
        print("---------"*4)
        print(f"weighted result is {pred}. unweighted result is {pred2}")
        print("the confidence distribution:")
        for i in range(len(confidences)):
            print(f"evidence{i}: {pred_labels[i]} with {confidences[i]}")


    outputs1.append(pred)
    outputs2.append(pred2)

print(n_correct, n_total)
print(n_correct / n_total)
print(total, corr, corr/total)

def result_print(targets, outputs):
    result_dict = {
        'f1_macro': f1_score(targets, outputs, zero_division=0, average='macro'),
        'Accuracy' : accuracy_score(targets, outputs, ),
        'f1':tuple(f1_score(targets, outputs,  zero_division=0, average=None)),
        'precision': tuple(precision_score(targets, outputs,  zero_division=0, average=None)),
        'recall': tuple(recall_score(targets, outputs,  zero_division=0, average=None)),
    }
    return result_dict
print("RESULT:")
print(f"--label prediction file:{args.label_prediction}")
print('--weigted result')
printf(result_print(targets, outputs1))
print('--no weigted result')
printf(result_print(targets, outputs2))