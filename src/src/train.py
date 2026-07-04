from preprocess import load_data
from preprocess import preprocess

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

import pandas as pd


# -----------------------
# Load Dataset
# -----------------------

df = load_data("../data/churn.csv")

X_train, X_test, y_train, y_test = preprocess(df)


# -----------------------
# Logistic Regression
# -----------------------

lr = LogisticRegression(max_iter=1000)

lr.fit(X_train, y_train)

lr_pred = lr.predict(X_test)

lr_prob = lr.predict_proba(X_test)[:, 1]


print("========== Logistic Regression ==========")

print("Accuracy :", accuracy_score(y_test, lr_pred))

print("Precision :", precision_score(y_test, lr_pred))

print("Recall :", recall_score(y_test, lr_pred))

print("F1 Score :", f1_score(y_test, lr_pred))

print("ROC AUC :", roc_auc_score(y_test, lr_prob))


# -----------------------
# Random Forest
# -----------------------

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

rf_prob = rf.predict_proba(X_test)[:, 1]


print("\n========== Random Forest ==========")

print("Accuracy :", accuracy_score(y_test, rf_pred))

print("Precision :", precision_score(y_test, rf_pred))

print("Recall :", recall_score(y_test, rf_pred))

print("F1 Score :", f1_score(y_test, rf_pred))

print("ROC AUC :", roc_auc_score(y_test, rf_prob))


# -----------------------
# Cross Validation
# -----------------------

encoder_df = pd.read_csv("../data/churn.csv")

if "customerID" in encoder_df.columns:
    encoder_df.drop("customerID", axis=1, inplace=True)

encoder_df["TotalCharges"] = pd.to_numeric(
    encoder_df["TotalCharges"],
    errors="coerce"
)

encoder_df.fillna(0, inplace=True)

from sklearn.preprocessing import LabelEncoder

enc = LabelEncoder()

for col in encoder_df.select_dtypes(include="object").columns:
    encoder_df[col] = enc.fit_transform(encoder_df[col])

X = encoder_df.drop("Churn", axis=1)

y = encoder_df["Churn"]

print("\nCross Validation")

print(
    "Logistic Regression:",
    cross_val_score(
        lr,
        X,
        y,
        cv=5,
        scoring="accuracy"
    ).mean()
)

print(
    "Random Forest:",
    cross_val_score(
        rf,
        X,
        y,
        cv=5,
        scoring="accuracy"
    ).mean()
)
