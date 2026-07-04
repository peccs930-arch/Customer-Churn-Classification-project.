import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


def evaluate_model(model_name, y_test, predictions, probabilities):

    print(f"\n{'='*50}")
    print(f"{model_name} Evaluation")
    print(f"{'='*50}")

    print("Accuracy :", round(accuracy_score(y_test, predictions), 4))
    print("Precision:", round(precision_score(y_test, predictions), 4))
    print("Recall   :", round(recall_score(y_test, predictions), 4))
    print("F1 Score :", round(f1_score(y_test, predictions), 4))
    print("ROC AUC  :", round(roc_auc_score(y_test, probabilities), 4))


def plot_confusion_matrix(y_test, predictions, model_name):

    cm = confusion_matrix(y_test, predictions)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm
    )

    disp.plot(cmap="Blues")

    plt.title(f"{model_name} - Confusion Matrix")

    plt.savefig(f"../images/{model_name}_confusion_matrix.png")

    plt.show()


def plot_roc_curve(model, X_test, y_test, model_name):

    RocCurveDisplay.from_estimator(
        model,
        X_test,
        y_test
    )

    plt.title(f"{model_name} - ROC Curve")

    plt.savefig(f"../images/{model_name}_roc_curve.png")

    plt.show()
