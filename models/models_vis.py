import scipy.sparse as sp
import pandas as pd
from evaluator_vis import TextModelEvaluator, evaluate_models
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

print(evaluate_models(evaluator, models, plot=True))