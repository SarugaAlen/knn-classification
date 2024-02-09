# K-Nearest Neighbors (KNN) Classifier

## Overview
This Python project implements the K-Nearest Neighbors (KNN) algorithm for classification. The KNN algorithm is a simple and effective machine learning algorithm used for both classification and regression tasks. The `Classifier.py` module contains the implementation of the KNN algorithm, while the `main.py` script demonstrates its usage on the Iris dataset.

## `Classifier.py` Module

### Class: `Classifier`
The `Classifier` class represents the KNN classifier. It includes methods for fitting the model, making predictions, performing k-fold cross-validation, calculating accuracy, visualizing instances, and writing results to a CSV file.

#### Constructor
- `num_neighbors`: Number of neighbors to consider in the KNN algorithm (default is 5).
- `distance_metric`: Distance metric used for calculating distances between instances ('euclidean' or 'manhattan', default is 'euclidean').

#### Methods
1. `fit(X, y)`: Fits the model with training data.
2. `predict(X_train)`: Predicts the labels for test instances.
3. `k_fold_cross_validation(X, y, num_folds)`: Performs k-fold cross-validation and prints average accuracy.
4. `accuracy(predict, y)`: Calculates and prints the accuracy of predictions.
5. `visualize_instance(instance_index)`: Visualizes a selected instance in a scatter plot.
6. `write_to_csv()`: Appends results to a CSV file.

## `main.py` Script

The `main.py` script demonstrates the usage of the KNN classifier on the Iris dataset. It includes the following steps:
1. Load Iris dataset using scikit-learn.
2. Create an instance of the `Classifier` class.
3. Normalize features using min-max normalization.
4. Split the dataset into training and testing sets.
5. Fit the model, make predictions, and calculate accuracy.
6. Visualize a selected instance in a scatter plot.
7. Perform k-fold cross-validation and write results to a CSV file.

## Usage
To run the project, execute the `main.py` script. Modify parameters such as the number of neighbors and distance metric in the script as needed.

```bash
python main.py 
```
## Dependencies

Ensure you have the required dependencies installed using the following:

```bash
pip install numpy scikit-learn seaborn matplotlib pandas
```
Feel free to customize the project and incorporate it into your machine learning workflows.