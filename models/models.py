import scipy.sparse as sp
import pandas as pd
from evaluator import TextModelEvaluator, evaluate_models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

#Load the previous treated data.
X = sp.load_npz("../data/Reviews_prepared_x.npz")
y = pd.read_csv("../data/Reviews_prepared_y.csv")["Label"].values

#Creates a new evaluator class, here you can change the test_size, randon_state and stratify values.
evaluator = TextModelEvaluator(X, y, test_size=0.2, random_state=42, stratify=True)
#Some models to test.
models = {
    "LogReg": LogisticRegression(max_iter=1000),
    "LinearSVC": LinearSVC(),
    "MultinomialNB": MultinomialNB()
}

table, relatory = evaluate_models(evaluator, models, average="binary")
print(table)                # ranking by F1.
print(relatory["LogReg"])  # classification_report of a specific model, in this case "LogReg".

# Cross-validation optional.
cv_res = evaluator.cross_validate(LogisticRegression(max_iter=1000), cv=5)
print(cv_res["cv_mean"])

# Grid search optional.
gs = evaluator.grid_search(
    LogisticRegression(max_iter=1000),
    param_grid={"C": [0.1, 1, 3], "penalty": ["l2"], "solver": ["liblinear", "lbfgs"]},
    cv=5, scoring="f1"
)
print(gs.best_params_, gs.best_score_)
