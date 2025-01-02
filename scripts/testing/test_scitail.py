import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from claim_verification.utils.Datasets import SciFactLabelPredictionDataset
from esim.data import NLIDataset

from scripts.training.train_scitail import add_knowledge_worker, trans_to_tensor
from scripts.testing.load_params import load_kbert_params, load_esim_params

from claim_verification.utils.evaluate import evaluate, oracle_evaluate
from claim_verification.utils.Functions import printf
from tqdm import tqdm



class SciTailDataset(Dataset):
    def __init__(self, file : str):
        self.samples = []

        label_encodings = {'entailment': 0, 'neutral': 1, 'contradiction': 2}

        with open(file, 'r', encoding='utf8') as input_data:
            next(input_data)

            for line in input_data:
                line = line.strip().split("\t")

                # Ignore sentences that have no gold label.
                if line[0] == "-":
                    continue

                label = line[0]
                premise = line[5]
                hypothesis = line[6]

                self.samples.append({'claim': hypothesis,'rationale':premise, 'label':label_encodings[label]})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

# esim
def predict_by_esim(model, preprocessor, batch):
    """
    :param model: trained ESIM model
    :param preprocessor: 预处理工具
    :param ids: id列表，对应p-h组的唯一索引
    :param premises: 前提
    :param hypotheses: 假设
    :param labels: 标签（必须为'entailment','contradiction','neutral'）
    """

    data = preprocessor.read_scifact(batch)
    transformed_data = preprocessor.transform_to_indices(data)
    dataset = NLIDataset(transformed_data)

    model.eval()
    device = model.device

    # Deactivate autograd for evaluation.
    with torch.no_grad():
        premises = dataset.data['premises'].to(device)
        premises_lengths = torch.tensor(dataset.premises_lengths)
        hypotheses = dataset.data['hypotheses'].to(device)
        hypotheses_lengths = torch.tensor(dataset.hypotheses_lengths)

        _, probs = model(premises,
                        premises_lengths,
                        hypotheses,
                        hypotheses_lengths)

        # out_classes = probs.max(dim=1).indices
        return probs

# kbert
def predict_by_kbert(args, model, batch, kg, vocab):
    """
    :param args: 一些常用参数
    :param dataset: 数据集，经过read_dataset后返回的，已注入知识的
    :return prediction: 3 labels的tensor
    """
    model.eval()

    with torch.no_grad():
        example = add_knowledge_worker(batch, kg, vocab, args)
        input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vms_batch = trans_to_tensor(example)
        vms_batch = torch.LongTensor(vms_batch)
        try:
            loss, logits = model(input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vms_batch)
        except:
            print(input_ids_batch)
            print(input_ids_batch.size())
            print(vms_batch)
            print(vms_batch.size())

        logits = nn.Softmax(dim=1)(logits)

        return logits


# kb_esim
def predict_by_kbesim(ensemble_func, kbert_path, esim_path, esim_config, dataset, kgname=None,batch_size=32):
    # For simplicity, we use DataParallel wrapper to use multiple GPUs.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------Kbert preparation--------------------
    print("start preparing kbert")
    kbert, vocab, kg, args = load_kbert_params(device, kbert_path, kgname)
    kbert = kbert.to(device)

    # --------------------ESIM preparation---------------------
    print("start preparing esim models")
    esim, preprocessor = load_esim_params(device, esim_path, esim_config)

    # ----------Preparing Tesing data----------
    print("start preparing data")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    with torch.no_grad():
        esim_outputs = []
        kbert_outputs = []
        average_outputs = []
        targets = []

        t = tqdm(dataloader)
        for i, batch in enumerate(t):
            esim_probs = predict_by_esim(esim, preprocessor, batch)
            kbert_probs = predict_by_kbert(args, kbert, batch, kg, vocab)
            average_probs = ensemble_func(esim_probs, kbert_probs)

            esim_pred = torch.argmax(esim_probs, dim=1)
            kbert_pred = torch.argmax(kbert_probs, dim=1)
            average_pred = torch.argmax(average_probs, dim=1)

            esim_outputs.extend(esim_pred.tolist())
            kbert_outputs.extend(kbert_pred.tolist())
            average_outputs.extend(average_pred.tolist())
            targets.extend(batch['label'].int().tolist())




def test(kbert_path, esim_path, esim_config, dataset, kgname='ConceptNetWithRank', batch_size=32):
    # For simplicity, we use DataParallel wrapper to use multiple GPUs.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------Kbert preparation--------------------
    print("start preparing kbert with {}".format(kgname))
    kbert, vocab, kg, args = load_kbert_params(device, kbert_path, kgname)
    kbert = kbert.to(device)

    # --------------------ESIM preparation---------------------
    print("start preparing esim models")
    esim, preprocessor = load_esim_params(device, esim_path, esim_config)

    # ----------Preparing Tesing data----------
    print("start preparing data")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    with torch.no_grad():
        esim_outputs = []
        kbert_outputs = []
        average_outputs = []
        targets = []

        t = tqdm(dataloader)
        for i, batch in enumerate(t):
            esim_probs = predict_by_esim(esim, preprocessor, batch)
            kbert_probs = predict_by_kbert(args, kbert, batch, kg, vocab)
            average_probs = (0.4*esim_probs+0.6*kbert_probs)/2

            esim_pred = torch.argmax(esim_probs, dim=1)
            kbert_pred = torch.argmax(kbert_probs, dim=1)
            average_pred = torch.argmax(average_probs, dim=1)

            esim_outputs.extend(esim_pred.tolist())
            kbert_outputs.extend(kbert_pred.tolist())
            average_outputs.extend(average_pred.tolist())
            targets.extend(batch['label'].int().tolist())

    print(kbert_outputs)
    kbert_result = evaluate(kbert_outputs, targets)
    esim_result = evaluate(esim_outputs, targets)
    kbesim_result = evaluate(average_outputs, targets)
    oracle_result = oracle_evaluate(kbert_outputs, esim_outputs, targets)

    return kbert_result, esim_result, kbesim_result, oracle_result

def case_study(kbert_path, esim_path, esim_config, dataset, kgname='ConceptNetWithRank', batch_size=32):
    # For simplicity, we use DataParallel wrapper to use multiple GPUs.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------Kbert preparation--------------------
    print("start preparing kbert with {}".format(kgname))
    kbert, vocab, kg, args = load_kbert_params(device, kbert_path, kgname)
    kbert = kbert.to(device)

    # --------------------ESIM preparation---------------------
    print("start preparing esim models")
    esim, preprocessor = load_esim_params(device, esim_path, esim_config)

    # ----------Preparing Tesing data----------
    print("start preparing data")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    with torch.no_grad():
        esim_outputs = []
        eprobs = []
        kbert_outputs = []
        kprobs = []
        average_outputs = []
        aprobs = []
        targets = []

        t = tqdm(dataloader)
        for i, batch in enumerate(t):
            esim_probs = predict_by_esim(esim, preprocessor, batch)
            kbert_probs = predict_by_kbert(args, kbert, batch, kg, vocab)
            average_probs = (0.4*esim_probs+0.6*kbert_probs)/2

            esim_pred = torch.argmax(esim_probs, dim=1)
            kbert_pred = torch.argmax(kbert_probs, dim=1)
            average_pred = torch.argmax(average_probs, dim=1)

            esim_outputs.extend(esim_pred.tolist())
            kbert_outputs.extend(kbert_pred.tolist())
            average_outputs.extend(average_pred.tolist())
            targets.extend(batch['label'].int().tolist())
            eprobs.extend(esim_probs.tolist())
            kprobs.extend(kbert_probs.tolist())
            aprobs.extend(average_probs.tolist())
            break

    with open("case_study.tsv", 'w', encoding='utf8') as f:
        f.write("\t".join(["premise","hypothesis","gold_label","esim_probs","esim_label","kbert_probs","kbert_label","kbesim_probs","kbesim_label"])+"\n")

        for i in range(len(targets)):
            premise = batch["rationale"][i]
            hypothesis = batch["claim"][i]
            gold_label = str(targets[i])
            esim_label = str(esim_outputs[i])
            kbert_label = str(kbert_outputs[i])
            kbesim_label = str(average_outputs[i])
            print(eprobs[i])
            eprob = f"{eprobs[i][0]},{eprobs[i][1]},{eprobs[i][2]}"
            kprob = f"{kprobs[i][0]},{kprobs[i][1]},{kprobs[i][2]}"
            aprob = f"{aprobs[i][0]},{aprobs[i][1]},{aprobs[i][2]}"
            f.write("\t".join([premise, hypothesis, gold_label, eprob, esim_label, kprob, kbert_label, aprob, kbesim_label])+"\n")


    # kbert_result = evaluate(kbert_outputs, targets)
    # esim_result = evaluate(esim_outputs, targets)
    # kbesim_result = evaluate(average_outputs, targets)
    # oracle_result = oracle_evaluate(kbert_outputs, esim_outputs, targets)
    #
    # return kbert_result, esim_result, kbesim_result, oracle_result




if __name__ == "__main__":

    kbert_path = "../../outputs/scitail_kbert/conceptnet.bin"
    esim_path = "../../outputs/scitail_esim/best.pth.tar"
    esim_config_path = "../../config/preprocessing/scitail_preprocessing.json"

    test_file = "../../datasets/scitail/scitail_test.txt"

    dataset = SciTailDataset(test_file)

    #case_study(kbert_path, esim_path, esim_config_path, dataset, kgname="ConceptNetWithRank")

    kbert_result, esim_result, kbesim_result, oracle_result = test(kbert_path, esim_path, esim_config_path, dataset, kgname=None)

    print("--KBERT result: ")
    printf(kbert_result)

    print("--ESIM result: ")
    printf(esim_result)

    print("--KB-ESIM result: ")
    printf(kbesim_result)

    print("--ORACLE result: ")
    print(f"oracle_score: {oracle_result}")