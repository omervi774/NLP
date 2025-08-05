import torch as tr
import torch
from torch.nn.functional import pad
import torch.nn as nn
import numpy as np
import loader as ld
from configs import *
from evaluation import accuracy


# select model to use
def train(hidden_size = 128, run_recurrent = True, use_RNN = True, num_epochs = 10, atten_size = 0):


    train_dataset, test_dataset, num_words, input_size = ld.get_data_set(batch_size)
    if run_recurrent:
        if use_RNN:
            model = ExRNN(input_size, output_size, hidden_size)
        else:
            model = ExGRU(input_size, output_size, hidden_size)
    else:
        if atten_size > 0:
            model = ExRestSelfAtten(input_size, output_size, hidden_size, atten_size)
        else:
            model = ExMLP(input_size, output_size, hidden_size)

    print("Using model: " + model.name())

    if reload_model:
        print("Reloading model")
        model.load_state_dict(torch.load("model_backups/" + model.name() + ".pth"))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_loss = 1.0
    test_loss = 1.0

    # training steps in which a test step is executed every test_interval
    loss_log_file = open("loss_backups/" + model.name() + "_loss.txt", "w")
    accuracy_log_file = open("accuracies_backups/" + model.name() + "_accuracy.txt", "w")

    for epoch in range(num_epochs):

        itr = 0 # iteration counter within each epoch

        for labels, reviews, reviews_text in train_dataset:   # getting training batches
        
            itr = itr + 1

            if (itr + 1) % test_interval == 0:
                test_iter = True
                labels, reviews, reviews_text = next(iter(test_dataset)) # get a test batch 
            else:
                test_iter = False

            # Recurrent nets (RNN/GRU)

            if run_recurrent:
                hidden_state = model.init_hidden(int(labels.shape[0]))

                for i in range(num_words):
                    output, hidden_state = model(reviews[:,i,:], hidden_state)  # HIDE

            else:  

            # Token-wise networks (MLP / MLP + Atten.) 
            
                sub_score = []
                if atten_size > 0:  
                    # MLP + atten
                    sub_score = model(reviews)
                else:               
                    # MLP
                    sub_score = model(reviews)

                output = torch.mean(sub_score, 1)
                
            # cross-entropy loss

            loss = criterion(output, labels)

            # optimize in training iterations

            if not test_iter:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # averaged losses
            if test_iter:
                test_loss = 0.8 * float(loss.detach()) + 0.2 * test_loss
            else:
                train_loss = 0.9 * float(loss.detach()) + 0.1 * train_loss

            if test_iter:
                log = (
                    f"Epoch [{epoch + 1}/{num_epochs}], "
                    f"Step [{itr + 1}/{len(train_dataset)}], "
                    f"Train Loss: {train_loss:.4f}, "
                    f"Test Loss: {test_loss:.4f}"
                )
                print(log)
                print(log, file=loss_log_file)

                # if not run_recurrent:
                    # nump_subs = sub_score.detach().numpy()
                    # labels = labels.detach().numpy()
                    # print_review(reviews_text[0], nump_subs[0,:,0], nump_subs[0,:,1], labels[0,0], labels[0,1])

                # saving the model
                # torch.save(model.state_dict(), "model_backups/" + model.name() + ".pth")
            
            print(itr, end="\r")
            if itr == len(train_dataset) :
                acc_train = accuracy(model, train_dataset, run_recurrent)
                acc_test = accuracy(model, test_dataset, run_recurrent)
                log = f"Epoch [{epoch + 1}/{num_epochs}], Train: {acc_train:.4f}, Test: {acc_test:.4f}"
                print(log)
                print(log, file=accuracy_log_file)
                torch.save(model.state_dict(), "model_backups/" + model.name() + ".pth")
                
    loss_log_file.close()
    accuracy_log_file.close()


def main():
    parameters = get_run_parameters()
    train(**parameters)


if __name__ == "__main__":
    main()