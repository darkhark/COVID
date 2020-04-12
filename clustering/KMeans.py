from sklearn.model_selection import train_test_split, KFold


def trainTestSplit(matrix, method="single", splits=3):
    X_Trains = []
    X_Tests = []
    if method == "single":
        X_train, X_test = train_test_split(matrix.toarray(), test_size=0.2, random_state=42)
        X_Trains.append(X_train)
        X_Tests.append(X_test)
    elif method == "kfolds":
        kf = KFold(splits, shuffle=True, random_state=42)
        for train_index, test_index in kf.split(matrix):
            X_Trains.append(matrix[train_index])
            X_Tests.append(matrix[test_index])
    return X_Trains, X_Tests




