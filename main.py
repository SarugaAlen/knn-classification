from Klasifikator import Classifier
from sklearn.datasets import load_iris

IrisData = load_iris(as_frame=True)

X = IrisData.data
y = IrisData.target

knn = Classifier(7, 'euclidean')

knn.fit(X, y)

predict = knn.predict(X)

correct_predictions = 0

for i, predicted, actual in zip(range(len(predict)), predict, y):
    print(f'Predicted: {predicted}, is really: {actual}')

    if predicted == actual:
        correct_predictions += 1

accuracy = (correct_predictions / len(predict)) * 100
print(f'\nAccuracy: {accuracy:.2f}%')

#knn.visualize_instance(133)