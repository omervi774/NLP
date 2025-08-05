# Sentiment Analysis with Neural Architectures

This project implements and compares multiple neural network architectures for **sentiment classification** using PyTorch. It supports classic MLPs, recurrent networks (RNN/GRU), and self-attention-enhanced MLPs.

## Architectures Supported

- **MLP** – Simple feedforward model
- **RNN** – Vanilla recurrent model
- **GRU** – Gated recurrent units
- **MLP with Restricted Self-Attention** – Combines token-wise MLP with local attention

## Features

- Configurable via command-line arguments
- Clean architecture and model abstraction
- Logging of training/testing loss and accuracy
- Visual evaluation of model predictions and attention weights
- Custom toy dataset support for debugging

## Project Structure

```
.
├── sentiment_start.py      # Entry point for training models
├── evaluation.py           # Evaluation, plotting, and error analysis
├── configs.py              # Model classes, config options, argument parsing
├── debug.py                # Debugging and inspection of toy data
├── loader.py               # (Required) Dataset loading logic (not included here)
├── model_backups/          # Folder for saved model weights
├── loss_backups/           # Folder for logging loss values
├── accuracies_backups/     # Folder for logging accuracy values
```

> ⚠️ You must provide a `loader.py` script that implements functions like `get_data_set()` and `get_toy_data()`.

## Usage

### Training

Run a model with default hyperparameters:

```bash
python sentiment_start.py MLP
```

With custom options:

```bash
python sentiment_start.py GRU --hidden_size 256 --num_epochs 20
```

### Evaluation

After training, run the evaluation script to generate accuracy/loss plots and view detailed predictions:

```bash
python evaluation.py GRU --hidden_size 256
```

### Debugging

Print the toy dataset:

```bash
python debug.py
```

## Requirements

- Python 3.7+
- PyTorch
- NumPy
- Pandas
- Matplotlib

Install with:

```bash
pip install torch numpy pandas matplotlib
```

## Notes

- Input data is assumed to be word-embedded reviews, where each review is a matrix of shape `[sequence_length × embedding_dim]`.
- Attention size (`--atten_size`) controls how wide the local neighborhood is in the restricted attention model.
- Log files are saved automatically under `loss_backups/` and `accuracies_backups/`.

## License

This project is for educational and research purposes only.
