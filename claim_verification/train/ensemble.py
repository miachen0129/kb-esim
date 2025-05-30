from torch.utils.data import DataLoader

from claim_verification.utils.Datasets import SciFactLabelPredictionDataset
from esim.data import NLIDataset

from claim_verification.train.train_kbert import add_knowledge_worker, trans_to_tensor
from claim_verification.test.load_params import load_kbert_params, load_esim_params
from claim_verification.test.kb_esim import predict_by_esim, predict_by_kbert
from claim_verification.utils.evaluate import evaluate, oracle_evaluate
from claim_verification.utils.Functions import printf

import os
import matplotlib.pyplot as plt

import time
import torch

import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score

from tqdm import tqdm
from esim.utils import correct_predictions


class MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size*2, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)  # 拼接两个输入张量
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        probs = nn.Softmax(dim=1)(x)
        return x, probs




def train(kbert_path, esim_path, esim_config, trainset, devset, kgname='ConceptNetWithRank', batch_size=32):
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
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(devset, batch_size=batch_size, shuffle=True )

    # ----------initialise MLP classifier----------
    print("start training MLP classifier")
    model = MLPClassifier(3, 64, 3)
    model.to(device)
    for name, param in model.named_parameters():
        print(name, param.size(), param.requires_grad)

    def train_evaluate(model,
                       dataloader,
                       optimizer,
                       criterion,
                       epoch_number,
                       max_gradient_norm):
        model.train()

        epoch_start = time.time()
        batch_time_avg = 0.0
        running_loss = 0.0
        correct_preds = 0

        outputs = []
        targets = []

        confusion = torch.zeros(3, 3, dtype=torch.long)

        tqdm_batch_iterator = tqdm(dataloader)
        for batch_index, batch in enumerate(tqdm_batch_iterator):
            batch_start = time.time()

            # Move input and output data to the GPU if it is used.
            X1 = predict_by_esim(esim, preprocessor, batch)
            X2 = predict_by_kbert(args, kbert, batch, kg, vocab)
            labels = batch['label'].long()

            optimizer.zero_grad()

            logits, probs = model(X1,X2)
            loss = criterion(logits, labels)
            loss.backward()

            pred = probs.argmax(dim=1)
            for j in range(pred.size()[0]):
                confusion[pred[j], labels[j]] += 1
            targets.extend(labels.tolist())
            outputs.extend(pred.tolist())

            nn.utils.clip_grad_norm_(model.parameters(), max_gradient_norm)
            optimizer.step()

            batch_time_avg += time.time() - batch_start
            running_loss += loss.item()
            correct_preds += correct_predictions(probs, labels)

            description = "Avg. batch proc. time: {:.4f}s, loss: {:.4f}" \
                .format(batch_time_avg / (batch_index + 1),
                        running_loss / (batch_index + 1))
            tqdm_batch_iterator.set_description(description)

        epoch_time = time.time() - epoch_start
        epoch_loss = running_loss / len(dataloader)
        epoch_accuracy = correct_preds / len(dataloader.dataset)
        result_dict = {
            'macro_f1': f1_score(targets, outputs, zero_division=0, average='macro'),
            'f1': tuple(f1_score(targets, outputs, zero_division=0, average=None)),
            'precision': tuple(precision_score(targets, outputs, zero_division=0, average=None)),
            'recall': tuple(recall_score(targets, outputs, zero_division=0, average=None)),
            'confusion': confusion
        }

        return epoch_time, epoch_loss, epoch_accuracy, result_dict

    def valid_evaluate(model, dataloader, criterion):
        model.eval()
        correct_preds = 0
        confusion = torch.zeros(3, 3, dtype=torch.long)

        outputs = []
        targets = []

        epoch_start = time.time()
        running_loss = 0.0
        running_accuracy = 0.0

        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataloader)):
                # Move input and output data to the GPU if it is used.
                X1 = predict_by_esim(esim, preprocessor, batch)
                X2 = predict_by_kbert(args, kbert, batch, kg, vocab)
                labels = batch['label'].long()

                optimizer.zero_grad()

                logits, probs = model(X1, X2)
                loss = criterion(logits, labels)
                #loss.backward()

                pred = probs.argmax(dim=1)
                for j in range(pred.size()[0]):
                    confusion[pred[j], labels[j]] += 1

                targets.extend(labels.tolist())
                outputs.extend(pred.tolist())

                running_loss += loss.item()
                correct_preds += correct_predictions(probs, labels)


        epoch_time = time.time() - epoch_start
        epoch_loss = running_loss / len(dataloader)
        epoch_accuracy = correct_preds / len(dataloader.dataset)
        result_dict = {
            'macro_f1': f1_score(targets, outputs, zero_division=0, average='macro'),
            'f1': tuple(f1_score(targets, outputs, zero_division=0, average=None)),
            'precision': tuple(precision_score(targets, outputs, zero_division=0, average=None)),
            'recall': tuple(recall_score(targets, outputs, zero_division=0, average=None)),
            'confusion': confusion
        }

        return epoch_time, epoch_loss, epoch_accuracy, result_dict

    # ----------train preparation----------
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode="max",
                                                           factor=0.5,
                                                           patience=0)



    best_score = 0.0
    start_epoch = 1
    epochs = 10

    # Data for loss curves plot.
    epochs_count = []
    train_losses = []
    valid_losses = []

    # Compute loss and accuracy before starting (or resuming) training.
    _, valid_loss, valid_accuracy, result_dict = valid_evaluate(model,
                                                                valid_loader,
                                                                criterion)
    print("\t* Validation loss before training: {:.4f}, accuracy: {:.4f}%, macro_f1: {:4f}%"
          .format(valid_loss, (valid_accuracy * 100), result_dict['macro_f1'] * 100))

    # -------------------- Training epochs ------------------- #
    print("\n",
          20 * "=",
          "Training ESIM models on device: {}".format(device),
          20 * "=")

    patience_counter = 0
    max_grad_norm = 10.0
    patience = 10
    target_dir = "../../outputs/mlp"
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    for epoch in range(start_epoch, epochs + 1):
        epochs_count.append(epoch)

        print("* Training epoch {}:".format(epoch))
        # epoch_time, epoch_loss, epoch_accuracy = train(models,
        #                                                train_loader,
        #                                                optimizer,
        #                                                criterion,
        #                                                epoch,
        #                                                max_grad_norm)
        epoch_time, epoch_loss, epoch_accuracy, result_dict = train_evaluate(model,
                                                                             train_loader,
                                                                             optimizer,
                                                                             criterion,
                                                                             epoch,
                                                                             max_grad_norm)

        train_losses.append(epoch_loss)
        print("-> Training time: {:.4f}s, loss = {:.4f}, accuracy: {:.4f}%"
              .format(epoch_time, epoch_loss, (epoch_accuracy * 100)))
        printf(result_dict)

        print("* Validation for epoch {}:".format(epoch))
        epoch_time, epoch_loss, epoch_accuracy, result_dict = valid_evaluate(model, valid_loader, criterion)
        # epoch_time, epoch_loss, epoch_accuracy = validate(models,
        #                                                   valid_loader,
        #                                                   criterion)
        macro_f1 = result_dict['macro_f1']

        valid_losses.append(epoch_loss)
        print("-> Valid. time: {:.4f}s, loss: {:.4f}, accuracy: {:.4f}%\n"
              .format(epoch_time, epoch_loss, (epoch_accuracy * 100)))
        printf(result_dict)

        # Update the optimizer's learning rate with the scheduler.
        scheduler.step(macro_f1)

        # Early stopping on validation accuracy.
        if macro_f1 < best_score:
            patience_counter += 1
        else:
            best_score = macro_f1
            patience_counter = 0
            # Save the best models. The optimizer is not saved to avoid having
            # a checkpoint file that is too heavy to be shared. To resume
            # training from the best models, use the 'esim_*.pth.tar'
            # checkpoints instead.
            torch.save({"epoch": epoch,
                        "models": model.state_dict(),
                        "best_score": best_score,
                        "epochs_count": epochs_count,
                        "train_losses": train_losses,
                        "valid_losses": valid_losses},
                       os.path.join(target_dir, "best.pth.tar"))

        # Save the models at each epoch.
        torch.save({"epoch": epoch,
                    "models": model.state_dict(),
                    "best_score": best_score,
                    "optimizer": optimizer.state_dict(),
                    "epochs_count": epochs_count,
                    "train_losses": train_losses,
                    "valid_losses": valid_losses},
                   os.path.join(target_dir, "esim_{}.pth.tar".format(epoch)))

        if patience_counter >= patience:
            print("-> Early stopping: patience limit reached, stopping...")
            print("best_score={}".format(best_score))
            break

    # Plotting of the loss curves for the train and validation sets.
    plt.figure()
    plt.plot(epochs_count, train_losses, "-r")
    plt.plot(epochs_count, valid_losses, "-b")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend(["Training loss", "Validation loss"])
    plt.title("Cross entropy loss")
    plt.show()

    return

if __name__ == "__main__":
    kbert_path = "../../outputs/scifact_with_noconcept/best_model.bin"
    esim_path = "../../data/checkpoints/scifact0424/best.pth.tar"
    esim_config_path = "../../config/scifact/scifact_preprocessing.json"

    train_file = "../../datasets/meta/sciFact/claims_train.jsonl"
    dev_file = "../../datasets/meta/sciFact/claims_dev.jsonl"
    corpus_file = "../../datasets/meta/sciFact/corpus.jsonl"

    trainset = SciFactLabelPredictionDataset(corpus_file, train_file)
    devset = SciFactLabelPredictionDataset(corpus_file, dev_file)

    train(kbert_path, esim_path, esim_config_path, trainset, devset, kgname='ConceptNetWithRank', batch_size=32)

