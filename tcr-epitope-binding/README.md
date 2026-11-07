# TCR–Epitope Binding Prediction

**Team:** Alessandro Querqui, Luca Nudo, Federico Trionfetti, Simone Mantero, Arman Salahshour – MSc in Data Science, Sapienza University of Rome, December 2024

**Goal:** predict whether a T-cell receptor (TCR) binds a given epitope from their amino-acid sequences. This is a binary classification task relevant to immunology and drug discovery.

## Data

[TCR–Epitope Binding](https://tdcommons.ai/multi_pred_tasks/tcrepitope/) task from the Therapeutic Data Commons (TDC): **47,190 TCR–epitope pairs** with epitope sequence, epitope chemical structure (SMILES), TCR sequence and a binding label (1 = binds, 0 = does not bind).

The dataset is downloaded directly in the notebook with the `PyTDC` package.

## Approach

- **Preprocessing:** sequences tokenised (amino acid → integer) and padded into a single feature matrix of 47,190 × 600.
- **Models:**
  - Random Forest (100 trees, max depth 10, stratified 5-fold cross-validation)
  - XGBoost (max depth 6, learning rate 0.1)
  - 1D CNN (4 convolutional layers, dropout, sigmoid output)

## Results

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| CNN | 78.25% | 74.50% | **85.44%** | **79.60%** |
| XGBoost | **78.34%** | **75.84%** | 77.99% | 76.90% |
| Random Forest | 74.09% | 73.42% | 75.10% | 74.25% |

XGBoost and the CNN have almost the same accuracy. The CNN reaches the best F1 thanks to a much higher recall, which matters when missing a real binding pair is costly. XGBoost is slightly more precise and much cheaper to train.

**Possible next steps:** protein language models (e.g. ProtBERT) and sequence-specific architectures.

## Files

- `[notebook_name].ipynb` – full notebook (Google Colab)
- `[report_name].pdf` – project report

## Tools

Python, scikit-learn, XGBoost, TensorFlow/Keras, PyTDC, pandas, NumPy, Matplotlib
