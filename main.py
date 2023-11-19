from Klasifikator import Classifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

IrisData = load_iris(as_frame=True)

X = IrisData.data
y = IrisData.target

knn = Classifier(8, 'manhattan') # manhattan euclidean

X_train, X_test, y_train, y_test = train_test_split(IrisData.data, IrisData.target, test_size=0.2) # random_state=42

knn.fit(X_train, y_train)

# knn.visualize_instance(133) # 133

predict = knn.predict(X_test)

knn.accuracy(predict, y)

knn.k_fold_cross_validation(X, y, 10)

knn.write_to_csv()



