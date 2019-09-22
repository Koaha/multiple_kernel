from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,mean_squared_log_error,classification_report
import pandas as pd
import numpy as np
from src.trees.tree_constructor import tree

if __name__ == "__main__":
    my_tree = tree(depth=3)
    my_tree.construct_tree()
    ls = my_tree.get_kernel_list()
    a = np.random.randint(1,10,12).reshape(3,4)
    b = np.random.randint(1,10,12).reshape(3,4)
    print(ls)
    for kernel in ls:
        k = kernel
        print(k(a,b))

    DATA_PATH = 'data/toyX.csv'
    LABEL_PATH = 'data/toyY.csv'

    readX = pd.read_csv(DATA_PATH)
    X = np.array(readX)
    readY = pd.read_csv(LABEL_PATH)
    Y = np.array(readY).reshape(-1,1)
    params = {
        'kernel': ls
    }
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=29)
    # SVC definition
    # model = NuSVC( decision_function_shape='ovo', class_weight=class_weights)
    model = SVC()
    clf = GridSearchCV(model, params, cv=2, verbose=1, n_jobs=2)
    print("Begin training SVM model...")
    clf.fit(X_train, y_train.reshape(-1))
    print("Finish training SVM.")
    # Perform prediction
    prediction = clf.predict(X_test)
    print(prediction)