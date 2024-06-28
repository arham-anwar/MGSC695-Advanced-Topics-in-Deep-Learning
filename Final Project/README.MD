
# Multi-class Text Classification with Transformers

## Project Overview

This project implements a multi-class text classification model using Transformer architecture on the 20 Newsgroups dataset. The model leverages a pre-trained GPT-2 tokenizer and custom Transformer blocks to achieve robust classification performance.

## Features

- **Model Architecture**: Utilizes a custom `TransformerClassifier` with multi-head attention and feedforward layers.
- **High Accuracy**: Achieved a test accuracy of 77.76%.
- **Efficient Training**: Implemented using PyTorch Lightning for streamlined training, validation, and evaluation.
- **Comprehensive Data Handling**: Custom dataset class for efficient text data processing.

## Setup

### Requirements

- Python 3.7+
- PyTorch Lightning
- Transformers (Hugging Face)
- scikit-learn
- Matplotlib
- Seaborn
- NumPy
- Rich

### Installation

Clone the repository and install the required packages:

\`\`\`bash
git clone https://github.com/your-username/transformer-text-classification.git
cd transformer-text-classification
pip install -r requirements.txt
\`\`\`

## Usage

### Data Loading & Exploration

Fetch the 20 Newsgroups dataset and explore basic statistics:

\`\`\`python
from sklearn.datasets import fetch_20newsgroups
data = fetch_20newsgroups(subset='all')
print(f"Number of samples: {len(data.data)}")
print(f"Number of categories: {len(data.target_names)}")
\`\`\`

### Tokenization

Initialize the GPT-2 tokenizer for text processing:

\`\`\`python
from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token
\`\`\`

### Model Training

Define and train the `TransformerClassifier` model:

\`\`\`python
from model import TransformerClassifier
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

model = TransformerClassifier(
    hidden_size=768, 
    num_classes=20, 
    max_seq_len=512, 
    n_heads=2, 
    n_layers=12, 
    lr=1e-5
).to(device)

trainer = Trainer(
    accelerator='gpu', 
    devices=1, 
    max_epochs=5, 
    callbacks=[ModelCheckpoint(monitor="val_acc", mode="max")]
)

trainer.fit(model, train_loader, val_loader)
\`\`\`

### Evaluation

Load the trained model and evaluate its performance:

\`\`\`python
model.load_state_dict(torch.load('path_to_checkpoint')['state_dict'])
trainer.test(model, test_loader)
\`\`\`

## Results

- **Test Accuracy**: 77.76%
- **Test Loss**: 0.97

The model demonstrated strong performance in classifying the diverse and complex documents within the 20 Newsgroups dataset.

## Conclusion

This project showcases the effective application of Transformer architecture for multi-class text classification. By leveraging advanced NLP techniques and efficient training frameworks, the model achieved significant accuracy and robustness.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [PyTorch Lightning](https://www.pytorchlightning.ai/)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [20 Newsgroups Dataset](http://qwone.com/~jason/20Newsgroups/)
