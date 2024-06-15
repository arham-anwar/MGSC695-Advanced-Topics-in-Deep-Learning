
# MGSC-695-075 – Advanced Topics in Management Science
## Assignments by Arham Anwar (ID: 261137773)

This repository contains three individual assignments for the MGSC-695-075 course. Each assignment focuses on different machine learning and deep learning techniques. Below is an overview of each assignment and its contents.

### Assignment I: Image Classification with CNNs
**Objective**: Apply various Convolutional Neural Network (CNN) techniques to classify images in the CIFAR-10 dataset.

**Contents**:
1. **Introduction**: Overview of the project and the models used.
2. **Simple CNN**:
   - **Architecture**:
     - Vanilla Basic CNN without Optuna Trials: A basic CNN architecture without hyperparameter optimization.
     - Simple CNN with Optuna Tuning: CNN architecture with hyperparameters optimized using Optuna.
     - Simple CNN with Custom Edits: Customized CNN architecture with additional layers and regularization techniques.
   - **Training Process**:
     - Optimizer: Adam optimizer with default parameters.
     - Learning Rate: Initially set to 0.001.
     - Regularization: Dropout with a rate of 0.5 applied to fully connected layers.
     - Number of Epochs: Trained for 50 epochs.
     - Data Augmentation: Applied random horizontal flipping and random cropping during training.
   - **Evaluation Results**: Confusion matrix, validation curve, and classification report.
3. **EfficientNet**:
   - **Architecture**: Implementation of EfficientNet from scratch, using mobile inverted bottleneck convolutional (MBConv) blocks and squeeze-and-excitation (SE) blocks.
   - **Training Process**: Utilized PyTorch Lightning for training, with a ReduceLROnPlateau scheduler for dynamic learning rate adjustment.
   - **Evaluation Results**: Performance metrics such as accuracy, precision, recall, and F1-score on the test dataset.
4. **MobileNet**:
   - **Architecture**: Implementation based on the SqueezeNet model, with modifications to the classifier layer for CIFAR-10 classification.
   - **Training Process**: Included data transformations, Adam optimizer, and learning rate scheduler.
   - **Evaluation Results**: Classification report, training, and validation performance curves.
5. **Result Documentation and Interpretation**: Comparison of model performances and learnings from the experiments.

**Files**:
- `Assignment_I_Simple_CNN.ipynb`
- `Assignment_I_EfficientNet.ipynb`
- `Assignment_I_MobileNet.ipynb`

### Assignment II: Text Generation with RNNs
**Objective**: Build and train Recurrent Neural Network (RNN) models to generate text in the style of Shakespeare.

**Contents**:
1. **Introduction**: Overview of text generation using RNNs.
2. **Seed for Repeatability**: Setting a random seed for reproducibility.
3. **Data Preparation**:
   - **Loading**: Downloading the "Tiny Shakespeare" dataset.
   - **Lower Casing the Data**: Standardizing text by converting to lowercase.
   - **Using 700K characters only**: Limiting the dataset to 700,000 characters for training.
   - **Character Dictionary**: Creating dictionaries for character-to-index and index-to-character mappings.
   - **Sequence Configuration**: Preparing training data for sequence prediction with specific sequence length and step size.
   - **Final Data Model Onboarding**: Defining a custom dataset class and creating DataLoader objects.
4. **Model Building**: Approaches using Keras and PyTorch Lightning, with a focus on the third model (punctuations removed).
5. **Model Hyperparameters**: Detailed explanation of hyperparameters like number of unique characters, hidden size, number of layers, learning rate, and dropout rate.
6. **Model Checkpoints and Best Weights**: Implementing ModelCheckpoint and EarlyStopping for optimal training.
7. **Text Generation**: Generating text with different temperature values to control randomness.
8. **Appendix**: Model outputs for different configurations.

**Files**:
- `Assignment_II_Text_Generation.ipynb`

### Assignment III: Multi-class Text Classification using Transformers
**Objective**: Train a Transformer model on the 20 Newsgroups dataset for text classification.

**Contents**:
1. **Introduction**: Overview of the project and the Transformer model.
2. **Setup**: Libraries and configurations.
3. **Hyperparameters & Device Configuration**: Details on batch size, sequence length, learning rate, and more.
4. **Data Loading & Exploration**: Fetching and exploring the 20 Newsgroups dataset.
5. **Tokenizer**: Initializing the GPT-2 tokenizer for text processing.
6. **Data Class**: Custom dataset class for handling text data.
7. **Data Loader**: Splitting data and creating data loaders.
8. **Attention Layer**: Implementation and benefits of the attention mechanism.
9. **Multihead Attention**: Enhancing feature extraction with multi-head attention.
10. **Projection Layer**: Feedforward network for processing attention outputs.
11. **Merging to Transformer Structure**: Combining components into a Transformer block.
12. **Classification Model (GPT2 Classifier)**: Custom neural network for text classification using GPT-2.

**Files**:
- `Assignment_III_Text_Classification.ipynb`

---

Each assignment is contained in its respective folder with the necessary Jupyter notebooks and related files. Detailed explanations and results are documented within each notebook. To run the code, ensure you have the required libraries installed and follow the instructions provided in each notebook.

---

Feel free to reach out if you have any questions or need further assistance.

---

Arham Anwar
MGSC-695-075
