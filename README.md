# Panther

Deep learning based tool to classify protein sequences into Structurally Similar Groups (5 Å distance) based on the CATH database.

# Dataset

The sequences were obtained from the CATH database (Version 4.3.0)

- Total Number of Sequences: 924,212
- Total Number of Classes: 6,396

## Preprocessing

1. Ran an All-vs-all BLAST scan to remove those sequences that had more than 90% sequence similarity but belonged to different superfamilies.
2. Removed those classes that had less than 100 datapoints. 

The resulting dataset has been described below:

- Number of Sequences: 788,287
- Number of Classes: 2,463

The sequences in this dataset were encoded into vector format using a number of techniques and then the performance was compared on each of them. The encoding techniques have been mentioned down below:

- BioVec (Implementation of the ProtVec)
- One-Hot Encoding
- ProtBert 

# Results

## BioVec Embedding Technique

| Model      | Accuracy | F1-Score | 
| ----------- | ----------- | ----------- |
| ANN   | 0.451 +- 0.0067%        | 0.439 +- 0.0063%        |
| CNN   | 0.395 +- 0.0026%        | 0.385 +- 0.0031%        |

## One-Hot Encoding

| Model      | Accuracy | F1-Score | 
| ----------- | ----------- | ----------- |
| ANN   | 0.596 +- 0.0187%        | 0.605 +- 0.0153%        |
| CNN   | 0.699 +- 0.0029%        | 0.697 +- 0.0031%        |

## ProtBert Embedding

| Model      | Accuracy | F1-Score | 
| ----------- | ----------- | ----------- |
| ANN   | 0.884 +- 0.0015%        | 0.883 +- 0.0015%        |
| CNN   | 0.874 +- 0.0017%        | 0.872 +- 0.0017%        |
