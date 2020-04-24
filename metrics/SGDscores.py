from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score
from sklearn.linear_model import SGDClassifier


def classificationScores(model_name, test, pred):
    print(model_name, ":\n")
    print("Accuracy Score: ", '{:,.3f}'.format(float(accuracy_score(test, pred)) * 100), "%")
    print("Precision:      ", '{:,.3f}'.format(float(precision_score(test, pred, average='macro')) * 100), "%")
    print("Recall:         ", '{:,.3f}'.format(float(recall_score(test, pred, average='macro')) * 100), "%")
    print("F1 score:       ", '{:,.3f}'.format(float(f1_score(test, pred, average='macro')) * 100), "%")
    print("\n")


def getTrainTestSplitScores(X, y_pred):
    X_train, X_test, y_train, y_test = train_test_split(X.toarray(), y_pred, test_size=0.2, random_state=42)

    print("X_train size:", len(X_train))
    print("X_test size:", len(X_test), "\n")

    sgd_clf = SGDClassifier(max_iter=10000, tol=1e-3, random_state=42)
    sgd_clf.fit(X_train, y_train)

    sgd_pred = cross_val_predict(sgd_clf, X_train, y_train, cv=3)
    classificationScores("Stochastic Gradient Descent Report (Training Set)", y_train, sgd_pred)
    sgd_pred = cross_val_predict(sgd_clf, X_test, y_test, cv=3)
    classificationScores("Stochastic Gradient Descent Report (Testing Set)", y_test, sgd_pred)

    sgd_cv_score = cross_val_score(sgd_clf, X.toarray(), y_pred, cv=10)
    print("Mean crossval Score - SGD: {:,.3f}".format(float(sgd_cv_score.mean()) * 100), "%")

