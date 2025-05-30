import os
import argparse
import pickle
import torch
import json

import matplotlib.pyplot as plt
import torch.nn as nn

from esim.data import Preprocessor

from torch.utils.data import DataLoader
from esim.data import NLIDataset
from esim.model import ESIM
from claim_verification.train.utils import train_evaluate, valid_evaluate
from claim_verification.utils.Functions import printf



def main(train_file,
         valid_file,
         embeddings_file,
         target_dir,
         hidden_size=300,
         dropout=0.5,
         num_classes=3,
         epochs=64,
         batch_size=32,
         lr=0.0004,
         patience=5,
         max_grad_norm=10.0,
         checkpoint=None):
    """
    Train the ESIM models on the SNLI datasets.

    Args:
        train_file: A path to some preprocessed data that must be used
            to train the models.
        valid_file: A path to some preprocessed data that must be used
            to validate the models.
        embeddings_file: A path to some preprocessed word embeddings that
            must be used to initialise the models.
        target_dir: The path to a directory where the trained models must
            be saved.
        hidden_size: The size of the hidden layers in the models. Defaults
            to 300.
        dropout: The dropout rate to use in the models. Defaults to 0.5.
        num_classes: The number of classes in the output of the models.
            Defaults to 3.
        epochs: The maximum number of epochs for training. Defaults to 64.
        batch_size: The size of the batches for training. Defaults to 32.
        lr: The learning rate for the optimizer. Defaults to 0.0004.
        patience: The patience to use for early stopping. Defaults to 5.
        checkpoint: A checkpoint from which to continue training. If None,
            training starts from scratch. Defaults to None.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(20 * "=", " Preparing for training ", 20 * "=")

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # -------------------- Data loading ------------------- #
    print("\t* Loading training data...{}".format(train_file))
    with open(train_file, "rb") as pkl:
        train_data = NLIDataset(pickle.load(pkl))
    print("successfully loaded")

    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)

    print("\t* Loading validation data...")
    with open(valid_file, "rb") as pkl:
        valid_data = NLIDataset(pickle.load(pkl))

    valid_loader = DataLoader(valid_data, shuffle=False, batch_size=batch_size)

    # -------------------- Model definition ------------------- #
    print("\t* Building models...")
    with open(embeddings_file, "rb") as pkl:
        embeddings = torch.tensor(pickle.load(pkl), dtype=torch.float)\
                     .to(device)

    model = ESIM(embeddings.shape[0],
                 embeddings.shape[1],
                 hidden_size,
                 embeddings=embeddings,
                 dropout=dropout,
                 num_classes=num_classes,
                 device=device).to(device)

    # -------------------- Preparation for training  ------------------- #
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode="max",
                                                           factor=0.5,
                                                           patience=0)

    best_score = 0.0
    start_epoch = 1

    # Data for loss curves plot.
    epochs_count = []
    train_losses = []
    valid_losses = []

    # Continuing training from a checkpoint if one was given as argument.
    if checkpoint:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint["epoch"] + 1
        best_score = checkpoint["best_score"]

        print("\t* Training will continue on existing models from epoch {}..."
              .format(start_epoch))

        model.load_state_dict(checkpoint["models"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        epochs_count = checkpoint["epochs_count"]
        train_losses = checkpoint["train_losses"]
        valid_losses = checkpoint["valid_losses"]

    # Compute loss and accuracy before starting (or resuming) training.
    _, valid_loss, valid_accuracy, result_dict = valid_evaluate(model,
                                             valid_loader,
                                             criterion)
    print("\t* Validation loss before training: {:.4f}, accuracy: {:.4f}%, macro_f1: {:4f}%"
          .format(valid_loss, (valid_accuracy*100), result_dict['macro_f1']*100))

    # -------------------- Training epochs ------------------- #
    print("\n",
          20 * "=",
          "Training ESIM models on device: {}".format(device),
          20 * "=")

    patience_counter = 0
    for epoch in range(start_epoch, epochs+1):
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
              .format(epoch_time, epoch_loss, (epoch_accuracy*100)))
        printf(result_dict)

        print("* Validation for epoch {}:".format(epoch))
        epoch_time, epoch_loss, epoch_accuracy, result_dict = valid_evaluate(model, valid_loader, criterion)
        # epoch_time, epoch_loss, epoch_accuracy = validate(models,
        #                                                   valid_loader,
        #                                                   criterion)
        macro_f1 = result_dict['macro_f1']

        printf(result_dict)

        valid_losses.append(epoch_loss)
        print("-> Valid. time: {:.4f}s, loss: {:.4f}, accuracy: {:.4f}%\n"
              .format(epoch_time, epoch_loss, (epoch_accuracy*100)))

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


if __name__ == "__main__":
    default_config = "../../config/scifact/scifact_training.json"

    parser = argparse.ArgumentParser(description="Train the ESIM models on SNLI")
    parser.add_argument("--config",
                        default=default_config,
                        help="Path to a json configuration file")
    parser.add_argument("--checkpoint",
                        default=None,
                        help="Path to a checkpoint file to resume training")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.realpath(__file__))

    if args.config == default_config:
        config_path = os.path.join(script_dir, args.config)
    else:
        config_path = args.config

    with open(os.path.normpath(config_path), 'r') as config_file:
        config = json.load(config_file)

    main("../../data/preprocessed/scifact_rationale/train_data.pkl",
         "../../data/preprocessed/scifact_rationale/dev_data.pkl",
         "../../data/preprocessed/scifact_rationale/embeddings.pkl",
         "../../data/checkpoints/scifact_rationale",
         config["hidden_size"],
         config["dropout"],
         config["num_classes"],
         config["epochs"],
         config["batch_size"],
         config["lr"],
         config["patience"],
         config["max_gradient_norm"],
         args.checkpoint)
