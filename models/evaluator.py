#dataclass is a shortcut by python to create a class that stores data, needing to write less code
from dataclasses import dataclass
#To note parameters and returns, no real change.
from typing import Dict, Any, Iterable, Union
#Treat numbers, arrays.
import numpy as np
#Treat tables, csv.
import pandas as pd
#Sparse matrix.
import scipy.sparse as sp
#Usefull classes for split data, validate and grid search.
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
#Classic metrics to evaluate models.
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, classification_report, confusion_matrix
)
#Define ArrayLike as a type.
ArrayLike = Union[np.ndarray, sp.spmatrix]

@dataclass
class TextModelEvaluator:
    #X = Features
    X: ArrayLike
    #Y = Labels
    y: Iterable
    #Test size, in this case, 20% of the dataset is used to test and 80% to train.
    test_size: float = 0.2
    #A fixed 'Seed' to create replicable results.
    random_state: int = 42
    #Mantain the proportions of classes equals in the train and test.
    stratify: bool = True

    #Especial method for the dataclass, separate in train and test data in initialization.
    def __post_init__(self):
        strat = self.y if self.stratify else None
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_state, stratify=strat
        )
    #Class to train and evaluate the models.
    #Average: "binary" for binary classifications; macro;weighted for multiclass.
    def evaluate(self, model, average: str = "binary") -> Dict[str, Any]:
        #Train the model with the correct data.
        model.fit(self.X_train, self.y_train)
        #Predict with the test x.
        y_pred = model.predict(self.X_test)
        #Calculates the accuracy, precision, recall and f1 score.
        acc = accuracy_score(self.y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            self.y_test, y_pred, average=average, zero_division=0
        )

        roc = None
        #Some models have a continuous score, so this calculates it they have predict_proba, decision_function.
        try:
            if hasattr(model, "predict_proba"):
                y_score = model.predict_proba(self.X_test)[:, 1]
            elif hasattr(model, "decision_function"):
                y_score = model.decision_function(self.X_test)
            else:
                y_score = None
            if y_score is not None and len(np.unique(self.y_test)) == 2:
                roc = roc_auc_score(self.y_test, y_score)
        except Exception:
            roc = None
        #Creates a confusion matrix with the test and the prediction.
        cm = confusion_matrix(self.y_test, y_pred)
        #Returns a dictionary with all the results.
        return {
            "model": model.__class__.__name__,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "roc_auc": roc,
            "confusion_matrix": cm,
            "classification_report": classification_report(self.y_test, y_pred, zero_division=0),
            "fitted_model": model,
        }
    #Method for cross validation.
    def cross_validate(self, model, cv: int = 5,
                       scoring=("accuracy", "precision", "recall", "f1"),
                       n_jobs: int = -1) -> Dict[str, Any]:
        res = cross_validate(model, self.X, self.y, cv=cv, scoring=scoring,
                             n_jobs=n_jobs, return_train_score=False)
        mean_scores = {k.replace("test_", ""): float(np.mean(v))
                       for k, v in res.items() if k.startswith("test_")}
        mean_scores["model"] = model.__class__.__name__
        return {"cv_raw": res, "cv_mean": mean_scores}
    #Method for grid search.
    def grid_search(self, model, param_grid: Dict[str, Any], cv: int = 5,
                    scoring: str = "f1", n_jobs: int = -1) -> GridSearchCV:
        gs = GridSearchCV(model, param_grid, cv=cv, scoring=scoring, n_jobs=n_jobs)
        gs.fit(self.X_train, self.y_train)
        return gs

#Compare all the models and give the better one, with higher scores.
def evaluate_models(evaluator: TextModelEvaluator, models: Dict[str, Any],
                    average: str = "binary") -> pd.DataFrame:
    rows, reports = [], {}
    for name, mdl in models.items():
        res = evaluator.evaluate(mdl, average=average)
        rows.append({
            "model": name,
            "accuracy": res["accuracy"],
            "precision": res["precision"],
            "recall": res["recall"],
            "f1": res["f1"],
            "roc_auc": res["roc_auc"],
        })
        reports[name] = res["classification_report"]
    table = pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)
    return table, reports
