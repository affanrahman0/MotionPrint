# MotionPrint

## Overview

This project aims to develop a unique identification system for individuals using data from accelerometers and gyroscopes. The system leverages machine learning models, data preprocessing, and ensemble techniques to achieve accurate identification.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Prerequisites](#prerequisites)
3. [Data Preparation](#data-preparation)
4. [Code Structure](#code-structure)
5. [Running the Code](#running-the-code)
6. [Results Interpretation](#results-interpretation)
7. [Further Improvement](#further-improvement)
8. [Authors](#authors)

## Getting Started

### Prerequisites

Before running the code, make sure you have the following prerequisites installed:

- Python 3.x
- Pandas
- Scikit-learn
- Imbalanced-learn (imblearn)

### Data Preparation
1. Collect accelerometer and gyroscope data of multiple individuals.
2. Run featureExtraction.py to extract the features from both acceleromer and gyroscope data seperately. It'll also add an label with it.
3. Ensure you have two CSV files: `accelerometer_data.csv` and `gyroscope_data.csv`. These files should contain your accelerometer and gyroscope data, respectively.
4. Your data should be structured such that each row represents a data sample, and the last column (label) contains the class information.
5. I have added 4 sample data of 4 individuals that can be use for feature extraction.

### Code Structure

- The code is organized into sections, each corresponding to a specific task. These tasks include data loading, data splitting, imputation, oversampling, classifier training, and ensemble creation.
- The code is divided into two main parts, one for accelerometer data and the other for gyroscope data. Both parts follow a similar structure.
- A Voting Classifier ensemble is created for both accelerometer and gyroscope data using individual classifiers.

## Running the Code

1. **Data Preparation**: Place your `accelerometer_data.csv` and `gyroscope_data.csv` files in the same directory as the script.

2. **Running the Script**: Run the script to execute the complete workflow. Ensure that you have the required libraries installed as mentioned in the prerequisites.

3. **Results**: The script will provide results for individual classifiers and ensemble models. It will display accuracy scores and classification reports.

## Results Interpretation

- The script evaluates individual classifiers for both accelerometer and gyroscope data and provides their respective accuracy scores and classification reports.

- Ensemble models (Voting Classifiers) are also evaluated, and their accuracy scores and classification reports are displayed.

## Further Improvement

To further improve the performance of the system, consider the following steps:

- Explore additional feature engineering techniques to create more informative features.

- Experiment with different machine learning algorithms or hyperparameter tuning to optimize model performance.

- Collect more data to increase the diversity and size of your dataset.

## Authors

[Affan Rahman]


