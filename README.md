# Decoder-Only Transformer From Scratch

A GPT-style decoder-only Transformer implemented and trained from scratch using PyTorch and the WikiText-2 dataset.

This project demonstrates the fundamental architecture and training pipeline behind autoregressive language models, including tokenization, causal self-attention, multi-head attention, feed-forward networks, backpropagation, and next-token text generation.

## Project Overview

The objective of this project was to understand how decoder-only Transformer language models work internally by implementing the architecture and training process without using pretrained model weights.

The GPT-2 tokenizer was used for tokenization, but the Transformer model itself was initialized and trained from scratch.

## Architecture

The model consists of the following components:

- Token Embedding
- Positional Embedding
- Multi-Head Causal Self-Attention
- Feed-Forward Neural Network
- Residual Connections
- Layer Normalization
- Language Modeling Head (LM Head)

### Model Configuration

| Parameter | Value |
|---|---:|
| Architecture | Decoder-Only Transformer |
| Number of Decoder Layers | 4 |
| Attention Heads | 4 |
| Embedding Dimension | 256 |
| Context Length | 512 |
| Vocabulary Size | 50,257 |
| Feed-Forward Dimension | 1,024 |
| Optimizer | AdamW |
| Learning Rate | 0.0003 |
| Training Iterations | 5,000 |

## Dataset

**WikiText-2 (Raw)**

Dataset: `Salesforce/wikitext`

Configuration: `wikitext-2-raw-v1`

The dataset was divided into three parts:

- Training tokens: 2,382,565
- Validation tokens: 246,275
- Test tokens: 282,071

## Training Process

The model was trained using next-token prediction.

The training pipeline includes:

1. Dataset loading and preprocessing
2. Text tokenization
3. Input and target sequence generation
4. Transformer forward propagation
5. Cross-entropy loss calculation
6. Backpropagation
7. AdamW optimizer updates
8. Validation loss evaluation
9. Autoregressive text generation

## Training Results

| Metric | Result |
|---|---:|
| Training Iterations | 5,000 |
| Final Training Loss | 4.3232 |
| Validation Loss | 5.6853 |
| Training Time | Approximately 28 minutes |

*Note: Training used randomly sampled batches from the training split. The complete training split was not systematically processed in one epoch.*

## Text Generation

The model generates text autoregressively by predicting one token at a time.

Given an initial prompt, it predicts the next token, appends that token to the sequence, and repeats the process.

## Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- Google Colab
- CUDA / NVIDIA GPU

## How to Run

1. Open the notebook in Google Colab.
2. Select a GPU runtime.
3. Install the required dependencies.
4. Execute the notebook cells sequentially.
5. Train the model.
6. Test autoregressive text generation with your own prompts.

## Key Learning Outcomes

Through this project, I explored:

- How decoder-only Transformers process sequential data.
- How causal masking prevents access to future tokens.
- How multi-head attention learns contextual relationships.
- How next-token prediction works.
- How backpropagation updates model parameters.
- How autoregressive language generation works.

## Future Improvements

- Systematic full-dataset training across multiple epochs.
- Training checkpoints and model recovery.
- Dropout and learning-rate scheduling.
- Perplexity-based evaluation.
- Temperature, Top-K and Top-P sampling.
- Architecture and hyperparameter optimization.

## Author

**AI & ML | Deep Learning | NLP | Generative AI**

This project was developed as a hands-on learning exercise to understand the internal mechanics of GPT-style language models.

