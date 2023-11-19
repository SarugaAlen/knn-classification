import csv
import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import euclidean, cityblock
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


class Classifier:
    def __init__(self, num_neighbors=5, distance_metric='euclidean'):
        self.X = None
        self.y = None
        self.num_neighbors = num_neighbors
        self.distance_metric = distance_metric
        self.accuracy_row = None
        self.avg_accuracy = None

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X_train):
        predictions = []

        for instance in range(len(X_train)):
            distances = []
            for i in range(len(self.X)):
                if self.distance_metric == 'euclidean':
                    distance_value = euclidean(X_train.iloc[instance, :], self.X.iloc[i, :])
                elif self.distance_metric == 'manhattan':
                    distance_value = cityblock(X_train.iloc[instance, :], self.X.iloc[i, :])
                else:
                    raise Exception('Unknown distance metric')

                distances.append((distance_value, self.y.iloc[i]))

            distances.sort(key=lambda x: x[0])

            neighbors = distances[:self.num_neighbors]

            class_counts = {}
            for _, label in neighbors:
                if label not in class_counts:
                    class_counts[label] = 1
                else:
                    class_counts[label] += 1

            predicted_class = max(class_counts, key=class_counts.get)
            predictions.append(predicted_class)

        return predictions

    def k_fold_cross_validation(self, X, y, num_folds=10):
        accuracy = []

        indices = [*range(len(X))]
        np.random.shuffle(indices)

        split_indices = np.array_split(indices, num_folds)

        for i in range(len(split_indices)):
            test_indices = split_indices[i]
            train_indices = [idx for num_folds in range(len(split_indices)) if num_folds != i for idx in
                             split_indices[num_folds]]

            train_X = X.iloc[train_indices]
            test_X = X.iloc[test_indices]
            train_y = y.iloc[train_indices]
            test_y = y.iloc[test_indices]

            train_X, test_X = pd.DataFrame(train_X, columns=X.columns), pd.DataFrame(test_X, columns=X.columns)
            train_y, test_y = pd.Series(train_y, name=y.name), pd.Series(test_y, name=y.name)

            self.fit(train_X, train_y)

            predictions = self.predict(test_X)

            fold_accuracy = accuracy_score(test_y, predictions) * 100
            print(f'Fold {i + 1} Accuracy: {fold_accuracy:.2f}%')
            accuracy.append(fold_accuracy)

        average_accuracy = np.mean(accuracy)
        print(f'Average Accuracy: {average_accuracy:.2f}%')
        self.accuracy_row = accuracy
        self.avg_accuracy = average_accuracy



    def accuracy(self, predict, y):
        correct_predictions = 0

        for i, predicted, actual in zip(range(len(predict)), predict, y):
            print(f'Predicted: {predicted}, is really: {actual}')

            if predicted == actual:
                correct_predictions += 1

        accuracy = (correct_predictions / len(predict)) * 100
        print(f'\nAccuracy: {accuracy:.2f}%')

    def visualize_instance(self, instance_index):
        podatki = load_iris(as_frame=True)

        X_izbrana = podatki.data.iloc[instance_index, :]
        y_izbrana = podatki.target.iloc[instance_index]

        X_ostali = podatki.data.drop(instance_index, axis=0)
        y_ostali = podatki.target.drop(instance_index)

        x_os, y_os = 0, 1

        sns.scatterplot(x=X_ostali.iloc[:, x_os],
                        y=X_ostali.iloc[:, y_os],
                        hue=podatki.target_names[y_ostali],
                        palette='colorblind')

        sns.scatterplot(x=[X_izbrana.iloc[x_os]],
                        y=[X_izbrana.iloc[y_os]],
                        hue=['Neznan'], style=['Neznan'],
                        markers={'Neznan': '^'})

        plt.show()

    def write_to_csv(self):
        filename = ''
        if self.distance_metric == 'euclidean':
            filename = 'Euclidean.csv'
        elif self.distance_metric == 'manhattan':
            filename = 'Manhattan.csv'

        with open(filename, 'a', newline='') as csvfile:
            fieldnames = ['Fold', 'NumNeighbors', 'DistanceMetric', 'Accuracy', 'AverageAccuracy']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if csvfile.tell() == 0:
                writer.writeheader()

            for i, accuracy in enumerate(self.accuracy_row):
                writer.writerow({'Fold': i + 1,
                                 'NumNeighbors': self.num_neighbors,
                                 'DistanceMetric': self.distance_metric,
                                 'Accuracy': accuracy})
            writer.writerow({'AverageAccuracy': self.avg_accuracy})
            writer.writerow({})

            print('Data appended to the file successfully')




