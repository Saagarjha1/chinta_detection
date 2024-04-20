# Stress Level Classification using Neural Networks

## Overview
This project aims to classify stress levels using heart rate data and neural networks. The dataset contains features extracted from heart rate signals, such as MEAN_RR, RMSSD, pNN25, pNN50, LF, HF, and LF_HF, along with their corresponding stress levels. The stress levels are categorized as 0 (low stress), 1 (medium stress), and 2 (high stress).

## Dataset
The dataset is provided in the `train.csv` file. It contains the following columns:
- MEAN_RR: Mean of RR intervals
- RMSSD: Root mean square of successive differences between RR intervals
- pNN25: Percentage of successive RR intervals differing by more than 25 ms
- pNN50: Percentage of successive RR intervals differing by more than 50 ms
- LF: Low-frequency power
- HF: High-frequency power
- LF_HF: Ratio of LF to HF
- stress_level: Target variable indicating the stress level (0, 1, or 2)

## Preprocessing
1. Missing values: Rows with missing values in the target label (`stress_level`) are dropped.
2. Invalid values: Rows with invalid stress levels (values other than 0, 1, or 2) are filtered out.
3. Standardization: Features are standardized using the StandardScaler to ensure all features have a mean of 0 and standard deviation of 1.
4. Train-test split: The dataset is split into training and testing sets using a 70-30 ratio.

## Model
- The neural network model consists of three hidden layers with ReLU activation functions.
- The output layer has a softmax activation function with three units corresponding to the three stress levels.
- Categorical crossentropy loss is used as the loss function, and the Adam optimizer is used for optimization.
- The model is compiled and trained on the training data for 50 epochs with a batch size of 1024.

## Evaluation
- The trained model is evaluated on the test data to measure its performance.
- The accuracy and loss metrics are plotted over epochs to visualize the training process.

## Usage
1. Ensure all necessary libraries are installed (`pandas`, `numpy`, `scikit-learn`, `tensorflow`, `matplotlib`).
2. Place the `train.csv` file containing the dataset in the project directory.
3. Run the provided Python script (`stress_level_classification.py`) to preprocess the data, train the model, and evaluate its performance.
4. The trained model (`stress_level_model.h5`) and the scaler object (`scaler.joblib`) are saved for future use.

## Files Included
- `train.csv`: Dataset containing heart rate data and stress levels.
- `stress_level_classification.py`: Python script for preprocessing, model training, and evaluation.
- `stress_level_model.h5`: Saved model file.
- `scaler.joblib`: Saved scaler object for feature standardization.
- `README.md`: Documentation providing an overview of the project and instructions for usage.

## Requirements
- Python 3
- pandas
- numpy
- scikit-learn
- tensorflow
- matplotlib

