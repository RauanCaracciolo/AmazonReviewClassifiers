import scipy.sparse as sp
import pandas as pd
from evaluator import TextModelEvaluator, evaluate_models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

#Load the previous treated data.
x_train = sp.load_npz('../data/Reviews_train_x.npz')
x_test = sp.load_npz('../data/Reviews_test_x.npz')
y_train = pd.read_csv('../data/Reviews_train_y.csv')["Label"].values
y_test = pd.read_csv('../data/Reviews_test_y.csv')["Label"].values

#Creates a new evaluator class.
evaluator = TextModelEvaluator(x_train, x_test, y_train, y_test)

#Some models to test.
models = {
    "LogReg": LogisticRegression(max_iter=1000),
    "LinearSVC": LinearSVC(),
    "MultinomialNB": MultinomialNB()
}

table, relatory = evaluate_models(evaluator, models, average="binary")
print(table)                # ranking by F1.
print(relatory["LogReg"])  # classification_report of a specific model, in this case "LogReg".
