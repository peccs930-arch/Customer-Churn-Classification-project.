import matplotlib.pyplot as plt
import pandas as pd


def plot_feature_importance(model, feature_names):

    if not hasattr(model, "feature_importances_"):
        print("Feature importance not available.")
        return

    importance = pd.Series(
        model.feature_importances_,
        index=feature_names
    )

    importance = importance.sort_values()

    plt.figure(figsize=(8,6))

    importance.plot(kind="barh")

    plt.title("Random Forest Feature Importance")

    plt.tight_layout()

    plt.savefig("../images/feature_importance.png")

    plt.show()


def compare_accuracy(results):

    models = list(results.keys())

    scores = list(results.values())

    plt.figure(figsize=(6,4))

    plt.bar(models, scores)

    plt.ylabel("Accuracy")

    plt.title("Model Accuracy Comparison")

    plt.savefig("../images/model_accuracy.png")

    plt.show()
