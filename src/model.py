import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from src.data import get_model_data
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

def run_baseline(X_train, X_test, y_train, y_test):
    num = X_train.select_dtypes("number").columns
    Xtr = X_train[num].fillna(0)
    Xte = X_test[num].fillna(0)

    clf = make_pipeline(
        StandardScaler(),                        # <-- đưa mọi cột về cùng thang đo
        LogisticRegression(max_iter=1000),
    )
    clf.fit(Xtr, y_train)
    p = clf.predict_proba(Xte)[:, 1]
    print(f"[Baseline LR] AUC={roc_auc_score(y_test, p):.3f} "
          f"PR-AUC={average_precision_score(y_test, p):.3f}")
    return clf


def time_split(X, y, thang):
    train_mask = thang.isin([3, 4])
    test_mask  = thang == 5
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test   = X[test_mask], y[test_mask]
    return X_train, X_test, y_train, y_test
    


def run_baseline(X_train, X_test, y_train, y_test):
    num = X_train.select_dtypes("number").columns        # chỉ lấy cột số
    Xtr = X_train[num].fillna(0)                          # Logistic không lay na
    Xte = X_test[num].fillna(0)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(Xtr, y_train)
    p = clf.predict_proba(Xte)[:, 1]                      # xác suất lớp "sống"
    print(f"[Baseline LR] AUC={roc_auc_score(y_test, p):.3f} "
          f"PR-AUC={average_precision_score(y_test, p):.3f}")
    return clf


if __name__ == "__main__":
    X, y, thang = get_model_data()
    X_train, X_test, y_train, y_test = time_split(X, y, thang)
    print("Train:", X_train.shape, "| Test:", X_test.shape)
    run_baseline(X_train, X_test, y_train, y_test)