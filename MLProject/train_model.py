from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score



def train_model1(name, model, X_train, y_train, X_test, y_test):
    model = model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print("------------------------------------------------------\n")
    print(name)
    print('r_a_s: ', roc_auc_score(y_test, pred))
    print('f1_score: ', f1_score(y_test, pred))
    print('Precinost: ', accuracy_score(y_test, pred))
    print('Odziv: ', recall_score(y_test, pred))
    print('Greska pri klasifikaciji: ', 1 - metrics.accuracy_score(y_test, pred))
    print("------------------------------------------------------\n")
    return model
