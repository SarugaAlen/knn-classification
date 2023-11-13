import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import euclidean, cityblock
import seaborn as sns
import matplotlib.pyplot as plt


class Classifier:
    def __init__(self, num_neighbors=5, distance_metric='euclidean'):
        self.num_neighbors = num_neighbors
        self.distance_metric = distance_metric

    def fit(self, X, y):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def predict(self, X):
        predictions = []

        for instance in range(len(X)):
            distances = []
            for i in range(len(self.X_train)):
                if self.distance_metric == 'euclidean':
                    distance_value = euclidean(X.iloc[instance, :], self.X_train.iloc[i, :])
                elif self.distance_metric == 'manhattan':
                    distance_value = cityblock(X.iloc[instance, :], self.X_train.iloc[i, :])
                else:
                    raise Exception('Neznana metrika razdalje')

                distances.append((distance_value, self.y_train.iloc[i]))

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


    def k_fold_cross_validation(self, X, y, k=10):
        test = []

    def visualize_instance(self, instance_index):
        podatki = load_iris(as_frame=True)

        X_izbrana = self.X_train.iloc[instance_index, :]
        y_izbrana = self.y_train.iloc[instance_index]

        X_ostali = self.X_train.drop(instance_index, axis=0)
        y_ostali = self.y_train.drop(instance_index)

        x_os, y_os = 0, 1

        sns.scatterplot(x=X_ostali.iloc[:, x_os],
                        y=X_ostali.iloc[:, y_os],
                        hue=podatki.target_names[y_ostali],
                        palette='colorblind')

        # Izrišemo eno izbrano instanco
        sns.scatterplot(x=[X_izbrana.iloc[x_os]],
                        y=[X_izbrana.iloc[y_os]],
                        hue=['Neznan'], style=['Neznan'],
                        markers={'Neznan': '^'})

        plt.show()
