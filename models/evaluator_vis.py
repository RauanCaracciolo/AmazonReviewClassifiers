from dataclasses import dataclass
from typing import Dict, Any, Iterable, Union
import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, classification_report, confusion_matrix, roc_curve
)
ArrayLike = Union[np.ndarray, sp.spmatrix]

@dataclass
class TextModelEvaluator:
    x: ArrayLike
    y: Iterable
    test_size: float = 0.2
    random_state: int = 42
    stratify: bool = True

    def __post_init__(self):
        strat = self.y if self.stratify else None
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.x, self.y, test_size=self.test_size, random_state=self.random_state
        )
    def evaluate(self, model, average: str = "binary", plot:bool = True):
        model.fit(self.x_train, self.y_train)
        y_pred = model.predict(self.x_test)

        acc = accuracy_score(self.y_test, y_pred)
        prec, rec, f1, _=precision_recall_fscore_support(
            self.y_test, y_pred, average= average, zero_division=0
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

        if plot:
            fig, axes = plt.subplots(1, 3 if roc else 2, figsize = (16,4))

            metrics = {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1" : f1}
            if roc is not None:
                metrics["ROC AUC"] = roc
            sns.barplot(x=list(metrics.keys()), y=list(metrics.values()), ax=axes[0], palette="Blues_d")
            axes[0].set_ylim(0,1)
            axes[0].set_title(f"Metrics - {model.__class__.__name__}")

            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[1])
            axes[1].set_title("Confusion Matrix")
            axes[1].set_xlabel("Predicted")
            axes[1].set_ylabel("True")

            if roc and y_score is not None:
                fpr, tpr, _ = roc_curve(self.y_test, y_score)
                axes[2].plot(fpr, tpr, label=f"AUC={roc:.2f}")
                axes[2].plot([0,1], [0,1], linestyle="--", color="gray")
                axes[2].set_xlabel("False Positive Rate")
                axes[2].set_ylabel("True Positive Rate")
                axes[2].set_title("ROC Curve")
                axes[2].legend()

            plt.tight_layout()
            plt.show()

            return{
                "model": model.__class__.__name__,
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1":f1,
                "roc_auc": roc,
                "confusion_matrix": cm,
                "classification_report": classification_report(self.y_test, y_pred, zero_division=0),
                "fitted_model": model,
            }
def evaluate_models(evaluator: TextModelEvaluator, models: Dict[str, Any],
                    average: str = "binary", plot: bool = False) -> pd.DataFrame:
    rows = []
    for name, mdl in models.items():
        res = evaluator.evaluate(mdl, average = average, plot = plot)
        rows.append({
            "model": name,
            "accuracy": res["accuracy"],
            "precision": res["precision"],
            "recall": res["recall"],
            "f1": res["f1"],
        })
    return pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)