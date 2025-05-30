import argparse
from claim_verification.train.train_kbert import BertClassifier
from brain.conceptnet import ConceptNet
from brain.conceptnet_with_rank import ConceptNetWithRank
from uer.model_builder import build_model
from uer.utils.config import load_hyperparam
from uer.utils.constants import *
from uer.utils.seed import set_seed
from uer.utils.vocab import Vocab
import torch

from esim.data import NLIDataset, Preprocessor
from esim.model import ESIM
import os
import pickle
import json


def load_esim_params(device,
                     esim_path,
                     default_config):

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
    parser = argparse.ArgumentParser(description="Preprocess the scifact datasets")
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
    print(f"label-dict:{config['labeldict']}")
    preprocessor = Preprocessor( lowercase=config["lowercase"],
                                 ignore_punctuation=config["ignore_punctuation"],
                                 num_words=config["num_words"],
                                 stopwords=config["stopwords"],
                                 labeldict=config["labeldict"],
                                 bos=config["bos"],
                                 eos=config["eos"]
                                )
    ## Load the word dict
    with open(config["worddict_file"],"rb") as file:
        worddict = pickle.load(file)
    preprocessor.worddict = worddict

    return model, preprocessor


def load_kbert_params(  device,
                        kbert_path,
                        kgname=None):
    """
    :param kbert_path: 训练好的kbert_classifer路径，bin文件
    :param columns: 数据集（tsv）的标题行，对应的index，dict
    """
    # ------------------Preparing args------------------
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Path options.
    parser.add_argument("--pretrained_model_path", type=str, default="../../models/google_model.bin", help="Path of the pretrained models.")
    # parser.add_argument("--output_model_path", required=True, type=str,
    #                     help="Path of the output models.")
    parser.add_argument("--vocab_path", default="../../models/google_vocab.txt", type=str,
                        help="Path of the vocabulary file.")

    # parser.add_argument("--train_path", type=str, required=True)
    # parser.add_argument("--dev_path", type=str, required=True)
    # parser.add_argument("--corpus_path", type=str, required=True)

    parser.add_argument("--config_path", default="../../models/google_config.json", type=str,
                        help="Path of the config file.")

    # Model options.
    parser.add_argument("--batch_size", type=int, default=8,
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
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="Learning rate.")
    parser.add_argument("--patience", type=float, default=10,
                        help="Learning rate.")
    parser.add_argument("--warmup", type=float, default=0.1,
                        help="Warm up value.")

    # Training options.
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="Dropout.")
    parser.add_argument("--epochs_num", type=int, default=64,
                        help="Number of epochs.")
    parser.add_argument("--report_steps", type=int, default=20,
                        help="Specific steps to print prompt.")
    parser.add_argument("--seed", type=int, default=7,
                        help="Random seed.")

    # Evaluation options.
    parser.add_argument("--mean_reciprocal_rank", action="store_true", help="Evaluation metrics for DBQA datasets.")

    # kg
    # parser.add_argument("--kg_name", required=True, help="KG name or path")
    parser.add_argument("--workers_num", type=int, default=1, help="number of process for loading datasets")
    parser.add_argument("--no_vm", action="store_true", help="Disable the visible_matrix")

    args = parser.parse_args()

    # Load the hyperparameters from the config file.
    args = load_hyperparam(args)

    # -------------------Preparing works--------------------
    ## Count the number of labels.
    labels_set = set((0,1,2))
    args.labels_num = len(labels_set)

    ## loading vocabulary
    vocab = Vocab()
    vocab.load(args.vocab_path)
    args.vocab = vocab

    ## loading kg
    # Build knowledge graph.
    if kgname=="ConceptNetWithRank":
        #kg_files = ['brain/kgs/conceptnet_1.tsv','brain/kgs/conceptnet_2.tsv']
        kg_files = [f"../../brain/scitail/conceptnet_{i}.tsv" for i in range(1,2)]
        kg = ConceptNetWithRank(kg_files=kg_files, vocab_path=args.vocab_path, predicate=True)
    elif kgname=="ConceptNet":
        kg_files = [f"../../brain/scitail/conceptnet_{i}.tsv" for i in range(1,2)]
        kg = ConceptNet(kg_files=kg_files, vocab_path=args.vocab_path, predicate=True)
    else:

        kg = ConceptNet(spo_files=[], vocab_path=args.vocab_path, predicate=True)


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