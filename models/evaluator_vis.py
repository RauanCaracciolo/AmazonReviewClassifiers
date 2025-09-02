from dataclasses import dataclass
from typing import Iterable, Union
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, confusion_matrix, roc_curve
)

ArrayLike = Union[np.ndarray, sp.spmatrix]


@dataclass
class TextModelEvaluator:
    x_train: ArrayLike
    x_test: ArrayLike
    y_train: Iterable
    y_test: Iterable



    def evaluate(self, model, average: str = "binary"):
        model.fit(self.x_train, self.y_train)
        y_pred = model.predict(self.x_test)

        acc = accuracy_score(self.y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            self.y_test, y_pred, average=average, zero_division=0
        )

        roc = None
        y_score = None
        try:
            if hasattr(model, "predict_proba"):
                y_score = model.predict_proba(self.x_test)[:, 1]
            elif hasattr(model, "decision_function"):
                y_score = model.decision_function(self.x_test)
            if y_score is not None and len(np.unique(self.y_test)) == 2:
                roc = roc_auc_score(self.y_test, y_score)
        except Exception:
            roc = None

        cm = confusion_matrix(self.y_test, y_pred)

        metrics = {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1}
        if roc is not None:
            metrics["ROC AUC"] = roc

        return {"model": model.__class__.__name__,
                "metrics": metrics,
                "confusion_matrix": cm,
                "roc": roc,
                "y_score": y_score}


def evaluate_models(evaluator, models, average="weighted", plot=False):
    results = []
    n_models = len(models)

    if plot:
        fig, axes = plt.subplots(n_models, 3, figsize=(16, 5 * n_models))
        if n_models == 1:  # caso especial para apenas um modelo
            axes = np.expand_dims(axes, axis=0)

    for idx, (name, mdl) in enumerate(models.items()):
        res = evaluator.evaluate(mdl, average=average)
        results.append(res)

        if plot:
            metrics = res["metrics"]
            cm = res["confusion_matrix"]
            roc = res["roc"]
            y_score = res["y_score"]

            sns.barplot(
                x=list(metrics.keys()),
                y=list(metrics.values()),
                hue=list(metrics.keys()),
                ax=axes[idx, 0],
                palette="Blues_d",
                legend=False
            )
            axes[idx, 0].set_ylim(0, 1)
            axes[idx, 0].set_title(f"Metrics - {res['model']}")

            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[idx, 1])
            axes[idx, 1].set_title("Confusion Matrix")
            axes[idx, 1].set_xlabel("Predicted")
            axes[idx, 1].set_ylabel("True")

            if roc and y_score is not None:
                fpr, tpr, _ = roc_curve(evaluator.y_test, y_score)
                axes[idx, 2].plot(fpr, tpr, label=f"AUC={roc:.2f}")
                axes[idx, 2].plot([0, 1], [0, 1], linestyle="--", color="gray")
                axes[idx, 2].set_xlabel("False Positive Rate")
                axes[idx, 2].set_ylabel("True Positive Rate")
                axes[idx, 2].set_title("ROC Curve")
                axes[idx, 2].legend()
            else:
                axes[idx, 2].axis("off")

    if plot:
        plt.subplots_adjust(wspace=0.3, hspace=0.6)
        plt.show()

    return results
