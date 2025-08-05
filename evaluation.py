import torch
import re
import matplotlib.pyplot as plt # type: ignore
import pandas as pd
from configs import *
import loader as ld

num_words = 100

def accuracy(model, loader, run_recurrent):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for labels, reviews, reviews_text in loader:
            if run_recurrent:
                hidden_state = model.init_hidden(int(labels.shape[0]))
                for i in range(num_words):
                    output, hidden_state = model(reviews[:,i,:], hidden_state)  # HIDE
            else:
                output =  torch.mean(model(reviews), 1)
            preds = torch.argmax(output, dim=1)
            labels = torch.argmax(labels,dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total     
    return accuracy


def plot_accuracy_from_log(model_name):
    filename = "accuracies_backups/" + model_name + "_accuracy.txt"
    epochs = []
    train_acc = []
    test_acc = []
    pattern = re.compile(r"Epoch \[(\d+)/\d+\], Train: ([\d.]+), Test: ([\d.]+)")

    # Read and parse the log file
    with open(filename, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                epoch = int(match.group(1))
                train = float(match.group(2))
                test = float(match.group(3))
                epochs.append(epoch)
                train_acc.append(train)
                test_acc.append(test)

    # Plotting
    # plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_acc, label='Train Accuracy', marker='o')
    plt.plot(epochs, test_acc, label='Test Accuracy', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f' {model_name} : Accuracy over Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



def plot_loss_from_log(model_name, per_epoch=False):
    filename = "loss_backups/" + model_name + "_loss.txt"
    train_losses = []
    test_losses = []
    epoch_markers = []
    
    with open(filename, 'r') as f:
        lines = f.readlines()
        
    # Regex to extract Epoch, Step, Train Loss, Test Loss
    pattern = re.compile(r"Epoch \[(\d+)/\d+\], Step \[(\d+)/\d+\], Train Loss: ([\d.]+), Test Loss: ([\d.]+)")
    
    for line in lines:
        match = pattern.search(line)
        if match:
            epoch = int(match.group(1))
            step = int(match.group(2))
            train = float(match.group(3))
            test = float(match.group(4))
            
            train_losses.append(train)
            test_losses.append(test)
            
            if step == 900:  # marker for last step of epoch
                epoch_markers.append(len(train_losses)-1)
    
    if per_epoch:
        # Aggregate by epoch using the last loss in each epoch (like your code's EMA does)
        train_losses = [train_losses[i] for i in epoch_markers]
        test_losses = [test_losses[i] for i in epoch_markers]
        x = list(range(1, len(train_losses)+1))  # epoch indices
        plt.xlabel("Epoch")
    else:
        x = list(range(1, len(train_losses)+1))  # iteration indices
        plt.xlabel("Iteration")

    plt.plot(x, train_losses, label="Train Loss")
    plt.plot(x, test_losses, label="Test Loss")
    plt.title(f" {model_name} : Loss over Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

cases = {
    (0,0) : "True positive ✅",
    (0,1) : "False positive",
    (1,0) : "False negative",
    (1,1) : "True negative"
}

def plot_word_scores(words, scores, label, pred):
    print("="*60)
    print(f'{cases[pred, label]} sample:')
    scores = scores.detach().numpy()

    review = " ".join(words)
    print(f'review : {review}')
    for word, score in zip(words, scores):
        print(f'{word} ({score[0]:.2f}, {score[1]:.2f})')
    
    avg_positive_score = np.mean(scores[:, 0])
    avg_negative_score = np.mean(scores[:, 1])
    print()
    print(f'mean score ({avg_positive_score:.2f}, {avg_negative_score:.2f})')


   
def error_analysis(model, loader, run_recurrent):
    if run_recurrent:
        return
    model.eval()
    tp_sample = tn_sample = fp_sample = fn_sample = None
    
    with torch.no_grad():
        for labels, reviews, reviews_text in loader:
            # reviews -> bs * 100(words) * 100(embedding)
            sub_scores = model(reviews) # bs * 100(words) * 2(p,n)
            output =  torch.mean(sub_scores, 1) # bs * 2(p,n)
            preds = torch.argmax(output, dim=1) # bs
            labels = torch.argmax(labels,dim=1) # bs
            
            for label, pred, sub_score, review_text in zip(labels, preds, sub_scores, reviews_text):
                label = label.item()
                pred = pred.item()

                if label == 0 and pred == 0 and tp_sample is None:
                    tp_sample = (review_text, sub_score, label, pred)
                elif label == 1 and pred == 1 and tn_sample is None:
                    tn_sample = (review_text, sub_score, label, pred)
                elif label == 1 and pred == 0 and fp_sample is None:
                    fp_sample = (review_text, sub_score, label, pred)
                elif label == 0 and pred == 1 and fn_sample is None:
                    fn_sample = (review_text, sub_score, label, pred)

                if all([tp_sample, tn_sample, fp_sample, fn_sample]):
                    break
            if all([tp_sample, tn_sample, fp_sample, fn_sample]):
                break
        for sample in [tp_sample, tn_sample, fp_sample, fn_sample]:
            plot_word_scores(*sample)

def plot_toy_data_set(model, run_recurrent):
    data = ld.get_toy_data(batch_size=1)
    for label, review, review_text in data:
        if run_recurrent:
                hidden_state = model.init_hidden(int(label.shape[0]))
                for i in range(num_words):
                    output, hidden_state = model(review[:,i,:], hidden_state)  # HIDE
        else:
            output =  torch.mean(model(review), 1)
            
        pred_class = torch.argmax(output, dim=1)
        label_class = torch.argmax(label, dim=1)

        # pre - formatting
        pred_label = 'positive 😃' if pred_class == 0 else 'negative 😞'
        true_label = 'positive 😃' if label_class == 0 else 'negative 😞'
        success = pred_class == label_class
        result_emoji = "🟢 Prediction Correct ✅" if success else "🔴 Prediction Wrong ❌"
        # plot_word_scores(review_text[0], model(review), 0 ,0)
        review_text = ' '.join(review_text[0])

        # Final output
        print("="*60)
        print(f"{result_emoji}")
        print(f"🧠 Model predicted: {pred_label}")
        print(f"🏷️  True label:     {true_label}")
        print("\n📝 Review:\n" + review_text)
    print("="*60)

        

def eval(hidden_size = 128, run_recurrent = True, use_RNN = True, num_epochs = 10, atten_size = 0):
    
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
    
    model.load_state_dict(torch.load("model_backups/" + model.name() + ".pth")) 

    error_analysis(model, test_dataset, run_recurrent)
    plot_accuracy_from_log(model.name())
    plot_loss_from_log(model.name(), True)
    # plot_toy_data_set(model, run_recurrent)
    
    
def main():
    parameters = get_run_parameters()
    eval(**parameters)

if __name__ == "__main__":
    main()


    
      

    