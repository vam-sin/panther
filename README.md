# Panther

Deep learning based tool to classify protein sequences into Structurally Similar Groups (5 Å distance) based on the CATH database.

# Dataset

The sequences were obtained from the CATH database (Version 4.2.0)

- Total Number of Sequences: 4,134,000
- Total Number of Classes: 5,515

## Preprocessing

Those classes that had less than 100 datapoints were removed from the dataset, the resulting dataset has been described below:

- Number of Sequences: 4,079,562
- Number of Classes: 3,898

The sequences in this dataset were encoded into vector format using a number of techniques and then the performance was compared on each of them. The encoding techniques have been mentioned down below:

- BioVec (Implementation of the ProtVec)
- One-Hot Encoding
- ProtBert (Working on this currently)

# Results

## BioVec Embedding Technique

| Model      | Accuracy | Sensitivity | 
| ----------- | ----------- | ----------- |
| ANN      | 65.88%       | 60.51%       |
| CNN   | 47.66%        | 31.60%        |
| Attention-BiLSTM   | 53.73%        | 43.05%        |

## One-Hot Encoding

| Model      | Accuracy | Sensitivity | 
| ----------- | ----------- | ----------- |
| CNN   | 76.83%        | 73.40%        |
| Residual CNN   |  76.73%       | 73.35%        |
| Residual Protein Net   | 74.76%        | 71.19%        |
