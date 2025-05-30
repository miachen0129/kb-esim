from multiprocessing import Pool
import torch
from torch import nn
import sys

from brain.conceptnet import ConceptNet
from brain.conceptnet_with_rank import ConceptNetWithRank
from uer.model_builder import build_model
from uer.utils.config import load_hyperparam
from uer.utils.constants import *
import argparse

from uer.utils.seed import set_seed
from uer.utils.vocab import Vocab
from scripts.training.train_scitail import BertClassifier


def add_knowledge_worker(params):
    p_id, sentences, columns, kg, vocab, args = params
    sentences_num = len(sentences)
    dataset = []
    for line_id, line in enumerate(sentences):
        if line_id % 1000 == 0:
            print("Progress of process {}: {}/{}".format(p_id, line_id, sentences_num))
            sys.stdout.flush()
        line = line.strip().split('\t')
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
    input_ids = torch.LongTensor([sample[0] for sample in dataset])[:100]
    label_ids = torch.LongTensor([sample[1] for sample in dataset])[:100]
    mask_ids = torch.LongTensor([sample[2] for sample in dataset])[:100]
    pos_ids = torch.LongTensor([example[3] for example in dataset])[:100]
    vms = torch.LongTensor([example[4] for example in dataset])[:100]

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
    # parser.add_argument("--train_path", type=str, required=True,
    #                     help="Path of the trainset.")
    # parser.add_argument("--dev_path", type=str, required=True,
    #                     help="Path of the devset.")
    # parser.add_argument("--test_path", type=str, required=True,
    #                     help="Path of the testset.")
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
    parser.add_argument("--kg_name", default="HowNet", help="KG name or path")
    parser.add_argument("--workers_num", type=int, default=1, help="number of process for loading datasets")
    parser.add_argument("--no_vm", action="store_true", help="Disable the visible_matrix")

    args = parser.parse_args()
    ## load the hyperparams from config file
    args = load_hyperparam(args)
    set_seed(args.seed)

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
        kg_files = ['brain/kgs/scifact_conceptnet.tsv','brain/kgs/conceptnet_2.tsv']
        kg = ConceptNetWithRank(kg_files=kg_files, vocab_path=args.vocab_path, predicate=True)
    else:
        kg_files = ['brain/kgs/scifact_conceptnet.tsv', 'brain/kgs/conceptnet_2.tsv']
        kg = ConceptNet(kg_files=kg_files, vocab_path=args.vocab_path, predicate=True)


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






def main():
    # For simplicity, we use DataParallel wrapper to use multiple GPUs.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    columns = {"text_a":1,"text_b":2,"label":0}
    kbert_path = "../../outputs/classifier_model.bin"

    model, vocab, kg, args = load_kbert_params(device, kbert_path, columns, kgname='HowNet',isRank=False)
    model = model.to(device)

    # preparing esim
    test_file = "../../datasets/snli/test.tsv"
    ids, sentences = [],[]
    with open(test_file, "r", encoding="utf8") as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            else:
                sentences.append(line)
                ids.append(str(i))

    test_dataset = read_dataset(sentences,
                                columns,
                                kg,
                                vocab,
                                args,
                                workers_num=args.workers_num
                                )

    ids = [item[0] for item in test_dataset][:100]
    test_dataset = [item[1:] for item in test_dataset][:100]
    print(ids)
    logits = predict_by_kbert(test_dataset, model)

    print(logits.size())
    pred1 = torch.argmax(logits, dim=1)
    label_ids = torch.LongTensor([sample[1] for sample in test_dataset])
    gold = label_ids
    print("the kbert accuracy:{}".format(torch.sum(pred1 == gold).item() / len(ids)))



if __name__ == "__main__":
    main()
