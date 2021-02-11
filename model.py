#Name: Valinda Vanam, ID:700703487
import itertools

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


class ML:
    def __init__(self):
        self.n_jobs = -1
        self.LR = LogisticRegression()
        self.SVM = SVC(probability=True)
        self.RF = RandomForestClassifier(n_estimators=1500)
        self.KNN = KNeighborsClassifier()
        self.MLP = MLPClassifier(hidden_layer_sizes=(1024, 1024,))
        self.GBC = GradientBoostingClassifier()
        self.DT = DecisionTreeClassifier(criterion="entropy")
        self.algorithms = [self.LR, self.SVM, self.RF, self.KNN, self.MLP, self.GBC, self.DT]
        self.names = ['Logistic Regression', 'SVM', 'Random Forest', 'K Neighbors Nneighbors', 'Multi Layer Perceptron',
                      'Gradient Boosting', 'Decision Tree']

    def train(self, xtrain, ytrain):
        for name, algorithm in zip(self.names, self.algorithms):
            print(f"{name} model is getting trained")
            algorithm.fit(xtrain, ytrain)

    def identify(self, xtest, ytest):
        predictedResult = list()
        accuracyScore = dict()
        rocScore = dict()
        confusionMatrix = dict()
        classificationReport = dict()
        for name, algorithm in zip(self.names, self.algorithms):
            print(f"{name} model is getting analyzed")
            y_pred = algorithm.predict(xtest)
            accuracyScore[name] = accuracy_score(ytest, y_pred)
            confusionMatrix[name] = confusion_matrix(ytest, y_pred)
            classificationReport[name] = classification_report(digits=6, y_true=ytest, y_pred=y_pred)
        predictedResult.extend([accuracyScore, confusionMatrix, classificationReport])
        return predictedResult


# confusion matrix
def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
