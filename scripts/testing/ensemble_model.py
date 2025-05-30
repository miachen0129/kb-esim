import pickle
import sys
import os
import json
import argparse

from multiprocessing import Pool
import torch
from torch import nn

from esim.data import NLIDataset, Preprocessor
from esim.model import ESIM


<<<<<<< Updated upstream
from brain.conceptnet import ConceptNet
=======
from brain import KnowledgeGraph
>>>>>>> Stashed changes
from brain.conceptnet_with_rank import ConceptNetWithRank
from uer.model_builder import build_model
from uer.utils.config import load_hyperparam
from uer.utils.constants import *
from uer.utils.seed import set_seed
from uer.utils.vocab import Vocab

<<<<<<< Updated upstream
from scripts.training.train_scitail import BertClassifier
=======
from run_kbert_cls import BertClassifier
import run_kbert_cls
>>>>>>> Stashed changes

# esim
def predict_by_esim(model, preprocessor, ids, premises, hypotheses, labels):
    """
    :param model: trained ESIM models
    :param preprocessor: 预处理工具
    :param ids: id列表，对应p-h组的唯一索引
    :param premises: 前提
    :param hypotheses: 假设
    :param labels: 标签（必须为'entailment','contradiction','neutral'）
    """
    data = {"ids":ids, "premises":premises, "hypotheses":hypotheses, "labels":labels}
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

def load_esim_params(device,
                     esim_path,
<<<<<<< Updated upstream
                     default_config="config/preprocessing/snli_preprocessing.json",
                     wordict_path=None):
=======
                     default_config="config/preprocessing/snli_preprocessing.json"):
>>>>>>> Stashed changes

    # ------------------Load The Model------------------
    checkpoint = torch.load(esim_path)
    ## Retrieving models parameters from checkpoint.
    vocab_size = checkpoint["models"]["_word_embedding.weight"].size(0)
    embedding_dim = checkpoint["models"]['_word_embedding.weight'].size(1)
    hidden_size = checkpoint["models"]["_projection.0.weight"].size(0)
    num_classes = checkpoint["models"]["_classification.4.weight"].size(0)
    ## Create and then load the models
    model = ESIM(vocab_size,
                embedding_dim,
                hidden_size,
                num_classes=num_classes,
                device=device).to(device)

    model.load_state_dict(checkpoint["models"])

    # ----------------Load the Preprocessor----------------
    ## load the preprocessor configs from json file
    parser = argparse.ArgumentParser(description="Preprocess the SNLI datasets")
    parser.add_argument(
        "--config",
        default=default_config,
        help="Path to a configuration file for preprocessing SciFact datasets"
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.realpath(__file__))

    if args.config == default_config:
        config_path = os.path.join(script_dir, args.config)
    else:
        config_path = args.config

    with open(os.path.normpath(config_path), "r") as cfg_file:
        config = json.load(cfg_file)

    ## Create the preprocessor according to the configs
    preprocessor = Preprocessor( lowercase=config["lowercase"],
                                 ignore_punctuation=config["ignore_punctuation"],
                                 num_words=config["num_words"],
                                 stopwords=config["stopwords"],
                                 labeldict=config["labeldict"],
                                 bos=config["bos"],
                                 eos=config["eos"]
                                )
    ## Load the word dict
    with open(wordict_path,"rb") as file:
        worddict = pickle.load(file)
    preprocessor.worddict = worddict

    return model, preprocessor

# kbert
def add_knowledge_worker(params):
    p_id, sentences, columns, kg, vocab, args = params
    sentences_num = len(sentences)
    dataset = []
    for line_id, line in enumerate(sentences):
        if line_id % 1000 == 0:
            print("Progress of process {}: {}/{}".format(p_id, line_id, sentences_num))
            sys.stdout.flush()
        #line = line.strip().split('\t')
        try:
            if len(line) == 3:
                label = int(line[columns["label"]])
                text = CLS_TOKEN + line[columns["text_a"]] + " " + SEP_TOKEN + " " + line[columns["text_b"]] + SEP_TOKEN
                tokens, pos, vm, _ = kg.add_knowledge_with_vm([text], add_pad=True, max_length=args.seq_length)

                tokens = tokens[0]
                pos = pos[0]
                vm = vm[0].astype("bool")
                token_ids = [vocab.get(t) for t in tokens]
                mask = []
                seg_tag = 1
                for t in tokens:
                    if t == PAD_TOKEN:
                        mask.append(0)
                    else:
                        mask.append(seg_tag)
                    if t == SEP_TOKEN:
                        seg_tag += 1

                dataset.append((str(line_id+1),token_ids, label, mask, pos, vm))

            elif len(line) == 4:  # for dbqa
                qid = int(line[columns["qid"]])
                label = int(line[columns["label"]])
                text_a, text_b = line[columns["text_a"]], line[columns["text_b"]]
                text = CLS_TOKEN + text_a + " " + SEP_TOKEN + " " + text_b + SEP_TOKEN

                tokens, pos, vm, _ = kg.add_knowledge_with_vm([text], add_pad=True, max_length=args.seq_length)
                tokens = tokens[0]
                pos = pos[0]
                vm = vm[0].astype("bool")

                token_ids = [vocab.get(t) for t in tokens]
                mask = []
                seg_tag = 1
                for t in tokens:
                    if t == PAD_TOKEN:
                        mask.append(0)
                    else:
                        mask.append(seg_tag)
                    if t == SEP_TOKEN:
                        seg_tag += 1

                dataset.append((str(line_id+1), token_ids, label, mask, pos, vm, qid))
            else:
                pass
        except:
            print("Error line: ", line)
    return dataset


def read_dataset(sentences, columns, kg, vocab, args, workers_num=1):
    print("Loading sentences....")
    sentence_num = len(sentences)

    print("There are {} sentence in total. We use {} processes to inject knowledge into sentences.".format(
        sentence_num, workers_num))
    if workers_num > 1:
        params = []
        sentence_per_block = int(sentence_num / workers_num) + 1
        for i in range(workers_num):
            params.append(
                (i, sentences[i * sentence_per_block: (i + 1) * sentence_per_block], columns, kg, vocab, args))
        pool = Pool(workers_num)
        res = pool.map(add_knowledge_worker, params)
        pool.close()
        pool.join()
        dataset = [sample for block in res for sample in block]
    else:
        params = (0, sentences, columns, kg, vocab, args)
        dataset = add_knowledge_worker(params)

    return dataset

def predict_by_kbert(dataset, model):
    """
    :param args: 一些常用参数
    :param dataset: 数据集，经过read_dataset后返回的，已注入知识的
    :return prediction: 3 labels的tensor
    """
    input_ids = torch.LongTensor([sample[0] for sample in dataset])
    label_ids = torch.LongTensor([sample[1] for sample in dataset])
    mask_ids = torch.LongTensor([sample[2] for sample in dataset])
    pos_ids = torch.LongTensor([example[3] for example in dataset])
    vms = torch.LongTensor([example[4] for example in dataset])

    model.eval()
    with torch.no_grad():
        try:
            loss, logits = model(input_ids, label_ids, mask_ids, pos_ids, vms)
            logits = nn.Softmax(dim=1)(logits)
            return logits
        except:
            print(input_ids)
            print(input_ids.size())
            print(vms)
            print(vms.size())

def load_kbert_params(  device,
                        kbert_path,
                        columns,
                        kgname=None,
                        isRank=True):
    """
    :param kbert_path: 训练好的kbert_classifer路径，bin文件
    :param columns: 数据集（tsv）的标题行，对应的index，dict
    """
    # ------------------Preparing args------------------
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Path options.
    parser.add_argument("--pretrained_model_path", default="./models/google_model.bin", type=str,
                        help="Path of the pretrained models.")
    parser.add_argument("--output_model_path", default="./outputs/classifier_model.bin", type=str,
                        help="Path of the output models.")
    parser.add_argument("--vocab_path", default="./models/google_vocab.txt", type=str,
                        help="Path of the vocabulary file.")
<<<<<<< Updated upstream

=======
    # parser.add_argument("--train_path", type=str, required=True,
    #                     help="Path of the trainset.")
    # parser.add_argument("--dev_path", type=str, required=True,
    #                     help="Path of the devset.")
    # parser.add_argument("--test_path", type=str, required=True,
    #                     help="Path of the testset.")
>>>>>>> Stashed changes
    parser.add_argument("--train_path", type=str, default="./datasets/snli/train.tsv")
    parser.add_argument("--dev_path", type=str, default="./datasets/snli/dev.tsv")
    parser.add_argument("--test_path", type=str, default="./datasets/snli/test.tsv")

    parser.add_argument("--config_path", default="./models/google_config.json", type=str,
                        help="Path of the config file.")

    # Model options.
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size.")
    parser.add_argument("--seq_length", type=int, default=256,
                        help="Sequence length.")
    parser.add_argument("--encoder", choices=["bert", "lstm", "gru", \
                                              "cnn", "gatedcnn", "attn", \
                                              "rcnn", "crnn", "gpt", "bilstm"], \
                        default="bert", help="Encoder type.")
    parser.add_argument("--bidirectional", action="store_true", help="Specific to recurrent models.")
    parser.add_argument("--pooling", choices=["mean", "max", "first", "last"], default="first",
                        help="Pooling type.")

    # Subword options.
    parser.add_argument("--subword_type", choices=["none", "char"], default="none",
                        help="Subword feature type.")
    parser.add_argument("--sub_vocab_path", type=str, default="models/sub_vocab.txt",
                        help="Path of the subword vocabulary file.")
    parser.add_argument("--subencoder", choices=["avg", "lstm", "gru", "cnn"], default="avg",
                        help="Subencoder type.")
    parser.add_argument("--sub_layers_num", type=int, default=2, help="The number of subencoder layers.")

    # Tokenizer options.
    parser.add_argument("--tokenizer", choices=["bert", "char", "word", "space"], default="bert",
                        help="Specify the tokenizer."
                             "Original Google BERT uses bert tokenizer on Chinese corpus."
                             "Char tokenizer segments sentences into characters."
                             "Word tokenizer supports online word segmentation based on jieba segmentor."
                             "Space tokenizer segments sentences into words according to space."
                        )

    # Optimizer options.
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate.")
    parser.add_argument("--warmup", type=float, default=0.1,
                        help="Warm up value.")

    # Training options.
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="Dropout.")
    parser.add_argument("--epochs_num", type=int, default=64,
                        help="Number of epochs.")
    parser.add_argument("--report_steps", type=int, default=100,
                        help="Specific steps to print prompt.")
    parser.add_argument("--seed", type=int, default=7,
                        help="Random seed.")

    # Evaluation options.
    parser.add_argument("--mean_reciprocal_rank", action="store_true", help="Evaluation metrics for DBQA datasets.")

    # kg
    parser.add_argument("--kg_name", default="ConceptNetWithRank", help="KG name or path")
    parser.add_argument("--workers_num", type=int, default=1, help="number of process for loading datasets")
    parser.add_argument("--no_vm", action="store_true", help="Disable the visible_matrix")

    args = parser.parse_args()
    ## load the hyperparams from config file
    args = load_hyperparam(args)

    # -------------------Preparing works--------------------
    ## Count the number of labels.
    labels_set = set((0,1,2))
    if columns == None:
        columns = {"text_a": 1, "text_b": 2, "label": 0}
    args.labels_num = len(labels_set)

    ## loading vocabulary
    vocab = Vocab()
    vocab.load(args.vocab_path)
    args.vocab = vocab

    ## loading kg
    # Build knowledge graph.
    if isRank:
<<<<<<< Updated upstream
        kg_files = ['brain/kgs/scifact_conceptnet.tsv','brain/kgs/conceptnet_2.tsv']
        kg = ConceptNetWithRank(kg_files=kg_files, vocab_path=args.vocab_path, predicate=True)
    else:
        kg_files = ['brain/kgs/scifact_conceptnet.tsv', 'brain/kgs/conceptnet_2.tsv']
        kg = ConceptNet(kg_files=kg_files, vocab_path=args.vocab_path, predicate=True)
=======
        #kg_files = ['brain/kgs/conceptnet_1.tsv','brain/kgs/conceptnet_2.tsv']
        kg_files = ['datasets/scifact/scifact_conceptnet.tsv']
        kg = ConceptNetWithRank(kg_files=kg_files, vocab_path=args.vocab_path, predicate=True)
    else:
        spo_files = [kgname]
        kg = KnowledgeGraph(spo_files=spo_files, vocab_path=args.vocab_path, predicate=True)
>>>>>>> Stashed changes


    # -------------------Preparing models--------------------
    ## Build bert models.
    ## A pseudo target is added.
    args.target = "bert"
    model = build_model(args)

    # Load or initialize parameters.
    if args.pretrained_model_path is not None:
        # Initialize with pretrained models.
        model.load_state_dict(torch.load(args.pretrained_model_path), strict=False)
    else:
        # Initialize with normal distribution.
        for n, p in list(model.named_parameters()):
            if 'gamma' not in n and 'beta' not in n:
                p.data.normal_(0, 0.02)

    # Build classification models.
    model = BertClassifier(args, model)

    ## Load trained models
    if kbert_path is not None:
        model.load_state_dict(torch.load(kbert_path,map_location=torch.device('cpu')), strict=False)
    elif args.output_model_path is not None:
        model.load_state_dict(torch.load(args.output_model_path,map_location=torch.device('cpu')), strict=False)
    else:
        print("no models available")
        return
    model = model.to(device)

    return model, vocab, kg, args


# Datset loader.
def batch_loader(ids, sentences, premises, hypotheses, labels, label_ids, batch_size):
    instances_num = len(ids)
    for i in range(instances_num // batch_size):
        ids_batch = ids[i * batch_size: (i + 1) * batch_size]
        sentences_batch = sentences[i * batch_size: (i + 1) * batch_size]
        premises_batch = premises[i * batch_size: (i + 1) * batch_size]
        hypotheses_batch = hypotheses[i * batch_size: (i + 1) * batch_size]
        labels_batch = labels[i * batch_size: (i + 1) * batch_size]
        label_ids_batch = label_ids[i * batch_size: (i + 1) * batch_size]
        yield ids_batch, sentences_batch, premises_batch,hypotheses_batch,labels_batch,label_ids_batch
    if instances_num > instances_num // batch_size * batch_size:
        ids_batch = ids[instances_num // batch_size * batch_size:]
        sentences_batch = sentences[instances_num // batch_size * batch_size:]
        premises_batch = premises[instances_num // batch_size * batch_size:]
        hypotheses_batch = hypotheses[instances_num // batch_size * batch_size:]
        labels_batch = labels[instances_num // batch_size * batch_size:]
        label_ids_batch = label_ids[instances_num // batch_size * batch_size:]
        yield ids_batch, sentences_batch, premises_batch,hypotheses_batch,labels_batch,label_ids_batch



def test_kbesim():
    # For simplicity, we use DataParallel wrapper to use multiple GPUs.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------Kbert preparation--------------------
    print("start preparing kbert")
    columns = {"text_a":1,"text_b":2,"label":0}
    kbert_path = "../../outputs/scifact_models/kbert2.bin"

    kbert, vocab, kg, args = load_kbert_params(device, kbert_path, columns, kgname='ConceptNetWithRank',isRank=True)
    kbert = kbert.to(device)

    # --------------------ESIM preparation---------------------
    print("start preparing esim models")
    #esim_path = "outputs/pretrained_esim.pth.tar"
    esim_path = "../../outputs/scifact_models/esim_2.pth.tar"
    default_config = "config/preprocessing/snli_preprocessing.json"
    #wordict_path = "datasets/preprocessed/worddict.pkl"
    wordict_path = "../../outputs/scifact_models/worddict.pkl"
    esim, preprocessor = load_esim_params(device, esim_path, default_config, wordict_path)
    reverse_labeldict = {'0':'entailment','1':'neutral','2':'contradiction'}
    label_dict = {'entailment':0,'neutral':1,'contradiction':2}

    # ----------Preparing Tesing data----------
    print("start preparing data")
    # test_file_esim = "./datasets/snli/snli_1.0_test.txt"
    # data = preprocessor.read_data(test_file_esim)
    test_file = "./datasets/scifact_2/rte_test.tsv"
    data = preprocessor.read_from_tsv(test_file)
    premises, hypotheses, labels = data['premises'], data['hypotheses'], data['labels']
    print(premises[0], hypotheses[0], labels[0])
    ids, sentences, label_ids = [],[],[]
    for i in range(len(premises)):
        ids.append(str(i+1))
        sentences.append([label_dict[labels[i]]," ".join(premises[i]), " ".join(hypotheses[i])])
        label_ids.append(label_dict[labels[i]])

    kbert_correct = 0
    # Confusion matrix.
    kbert_confusion = torch.zeros(args.labels_num, args.labels_num, dtype=torch.long)

    esim_correct = 0
    # Confusion matrix.
    esim_confusion = torch.zeros(args.labels_num, args.labels_num, dtype=torch.long)

    correct = 0
    oracle_correct = 0
    # Confusion matrix.
    confusion = torch.zeros(args.labels_num, args.labels_num, dtype=torch.long)

    for i, (ids_batch, sentences_batch, premises_batch,hypotheses_batch, labels_batch, label_ids_batch) \
        in enumerate(batch_loader(ids, sentences, premises, hypotheses, labels, label_ids, 100)):
        print("-"*10,"Batch {}".format(i+1),"-"*10)
        # predict by kbert
        kb_dataset = read_dataset(sentences_batch, columns, kg, vocab, args, workers_num=args.workers_num)
        kb_ids = [item[0] for item in kb_dataset]
        kb_dataset = [item[1:] for item in kb_dataset]
        logits = predict_by_kbert(kb_dataset, kbert)

        pred1 = torch.argmax(logits, dim=1)

        # predict by esim
        probs = predict_by_esim(esim, preprocessor, ids_batch,premises_batch, hypotheses_batch, labels_batch)
        pred2 = torch.argmax(probs, dim=1)

        # 求平均
        if logits.size() == probs.size():
            avg_prob = (logits+probs)/2
        else:
            avg_prob = torch.zeros(probs.size())
            for j, kb_id in enumerate(kb_ids):
                avg_prob[int(kb_id)-1] = (logits[j]+probs[int(kb_id)-1])/2

        avg_pred = torch.argmax(avg_prob, dim=1)
        gold = torch.tensor(label_ids_batch)

        for j in range(pred2.size()[0]):
            if pred2[j] == gold[j] or pred1[j] == gold[j]:
                oracle_correct += 1
        for j in range(avg_pred.size()[0]):
            confusion[avg_pred[j], gold[j]] += 1
            kbert_confusion[pred1[j], gold[j]] += 1
            esim_confusion[pred2[j], gold[j]] += 1
        correct += torch.sum(avg_pred == gold).item()

        esim_correct += torch.sum(pred2 == gold).item()
        print("the esim accuracy:{}".format(torch.sum(pred2 == gold).item() / len(ids_batch)))
        kbert_correct += torch.sum(pred1 == gold).item()
        print("the kbert accuracy:{}".format(torch.sum(pred1 == gold).item() / len(ids_batch)))
        print("oracle accuracy:{}".format(oracle_correct/(len(ids_batch)*(i+1))))
        print("the average accuracy:{}".format(torch.sum(avg_pred == gold).item() / len(ids_batch)))

    print("-" * 10, "kbert result", "-" * 10)
    print("Confusion matrix of KBERT:")
    print(kbert_confusion)
    print("Report precision, recall, and f1:")
    for i in range(kbert_confusion.size()[0]):
        p = kbert_confusion[i, i].item() / kbert_confusion[i, :].sum().item()
        r = kbert_confusion[i, i].item() / kbert_confusion[:, i].sum().item()
        f1 = 2 * p * r / (p + r)
        if i == 1:
            label_1_f1 = f1
        print("Label {}: {:.3f}, {:.3f}, {:.3f}".format(i, p, r, f1))
    print("Acc. (Correct/Total): {:.4f} ({}/{}) ".format(kbert_correct / len(ids), kbert_correct, len(ids)))


    print("-" * 10, "esim result", "-" * 10)
    print("Confusion matrix of ESIM:")
    print(esim_confusion)
    print("Report precision, recall, and f1:")
    for i in range(esim_confusion.size()[0]):
        p = esim_confusion[i, i].item() / esim_confusion[i, :].sum().item()
        r = esim_confusion[i, i].item() / esim_confusion[:, i].sum().item()
        f1 = 2 * p * r / (p + r)
        if i == 1:
            label_1_f1 = f1
        print("Label {}: {:.3f}, {:.3f}, {:.3f}".format(i, p, r, f1))
    print("Acc. (Correct/Total): {:.4f} ({}/{}) ".format(esim_correct / len(ids), esim_correct, len(ids)))

    print("-" * 10, "final result", "-" * 10)
    print("Confusion matrix of overall models:")
    print(confusion)
    print("Report precision, recall, and f1:")
    for i in range(confusion.size()[0]):
        p = confusion[i, i].item() / confusion[i, :].sum().item()
        r = confusion[i, i].item() / confusion[:, i].sum().item()
        f1 = 2 * p * r / (p + r)
        if i == 1:
            label_1_f1 = f1
        print("Label {}: {:.3f}, {:.3f}, {:.3f}".format(i, p, r, f1))
    print("Acc. (Correct/Total): {:.4f} ({}/{}) ".format(correct / len(ids), correct, len(ids)))
    print("Oracle. (OracleCorrect/Total): {:.4f} ({}/{}) ".format(oracle_correct / len(ids), oracle_correct, len(ids)))


<<<<<<< Updated upstream
=======




>>>>>>> Stashed changes
if __name__ == "__main__":
    test_kbesim()
