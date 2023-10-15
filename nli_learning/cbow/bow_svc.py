import numpy as np
import fasttext.util
from sklearn.utils import compute_sample_weight
from xgboost import XGBClassifier
from sklearn import metrics
from sklearn.svm import LinearSVC, NuSVC
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression

fasttext.util.download_model("ro", if_exists="ignore")  # English
ft = fasttext.load_model("cc.ro.300.bin")

WORD_EMBEDDING_DIM = 300

fasttext.util.reduce_model(ft, WORD_EMBEDDING_DIM)

np.seterr(divide="raise", invalid="raise")


def get_avg_bow(sentence: str) -> list[float]:
    """
    Returns the bag of words representation of a sentence
    """

    avg_vec = np.zeros(WORD_EMBEDDING_DIM)

    items = 0

    for word in sentence.split():
        avg_vec += ft.get_word_vector(word)
        items += 1

    return np.divide(avg_vec, items)


def transform_dataset_to_bow(dataset):
    inputs = []
    labels = []

    for item in dataset:
        all_sentence_1 = item[1]
        all_sentence_2 = item[2]
        all_labels = item[3]

        for sentence_1, sentence_2, label in zip(
            all_sentence_1, all_sentence_2, all_labels
        ):
            sentence_1_bow = get_avg_bow(sentence_1)
            sentence_2_bow = get_avg_bow(sentence_2)
            input_bow = np.concatenate((sentence_1_bow, sentence_2_bow), axis=0)
            inputs.append(input_bow)
            labels.append(label)

    return np.array(inputs), np.array(labels)


def train_svc(
    train_input,
    train_labels,
    validation_input,
    validation_labels,
    test_inputs,
    test_labels,
):
    """
    param_grid = {
        'C': [0.1, 0.5, 1, 2, 50],
        'class_weight': ['balanced'],
        'max_iter': [2500],
        'tol': [1e-5],
        'verbose': [1]
    }
    """
    param_grid_svc = {
        "C": [0.5],
        "class_weight": ["balanced"],
        "max_iter": [2500],
        "tol": [1e-05],
        "verbose": [1],
    }

    model_svc = LinearSVC()
    clf = GridSearchCV(model_svc, param_grid_svc, error_score="raise")
    clf.fit(train_input, train_labels)
    print(metrics.classification_report(test_labels, clf.predict(test_inputs)))


def train_logreg(
    train_input,
    train_labels,
    validation_input,
    validation_labels,
    test_inputs,
    test_labels,
):
    param_grid_svc = {
        "C": [0.01, 0.1, 1, 10, 100],
        "tol": [1e-3, 1e-2, 1e-1],
        "max_iter": [100, 500, 1000],
    }

    model_svc = LogisticRegression()
    clf = GridSearchCV(model_svc, param_grid_svc, error_score="raise")
    clf.fit(train_input, train_labels)
    print(metrics.classification_report(test_labels, clf.predict(test_inputs)))
    # Get the best hyperparameters
    best_params = clf.best_params_
    print("Best hyperparameters:", best_params)


def train_xgb(
    train_input,
    train_labels,
    validation_input,
    validation_labels,
    test_inputs,
    test_labels,
):
    # param_grid_xgb = {
    #     'min_child_weight': [1, 5],
    #     'gamma': [0.5,1.0],
    #     'subsample': [0.8, 1.0],
    #     'colsample_bytree': [1.0],
    #     'max_depth': [3, 4, 5],
    # }

    param_grid_xgb = {
        "colsample_bytree": [1.0],
        "gamma": [0.5],
        "max_depth": [5],
        "min_child_weight": [1],
        "subsample": [1.0],
    }

    model_xgb = XGBClassifier()
    clf = GridSearchCV(model_xgb, param_grid_xgb, error_score="raise")
    clf.fit(train_input, train_labels)
    print(metrics.classification_report(test_labels, clf.predict(test_inputs)))
