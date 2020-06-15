import os
import warnings
import shutil

import pandas as pd
import numpy as np
from itertools import cycle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import enet_path
from sklearn import datasets

# Import mlflow
import mlflow
import mlflow.sklearn


def plot_enet_descent_path(X, y, l1_ratio):
    # Compute paths
    eps = 5e-3  # the smaller it is the longer is the path

    # Reference the global image variable
    global image

    print("Computing regularization path using ElasticNet.")
    alphas_enet, coefs_enet, _ = enet_path(
        X, y, eps=eps, l1_ratio=l1_ratio, fit_intercept=False)

    # Display results
    fig = plt.figure(1)
    plt.gca()

    colors = cycle(['b', 'r', 'g', 'c', 'k'])
    neg_log_alphas_enet = -np.log10(alphas_enet)
    for coef_e, c in zip(coefs_enet, colors):
        plt.plot(neg_log_alphas_enet, coef_e, linestyle='--', c=c)

    plt.xlabel('-Log(alpha)')
    plt.ylabel('coefficients')
    title = 'ElasticNet Path by alpha for l1_ratio = ' + str(l1_ratio)
    plt.title(title)
    plt.axis('tight')

    # Display images
    image = fig

    # Save figure
    fig.savefig("out/ElasticNet-paths_l1_ratio_{}_.png".format(str(l1_ratio)))

    # Close plot
    plt.close(fig)

    # Return images
    return image


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


def train_diabetes(data, in_alpha, in_l1_ratio):
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    mlflow.set_experiment("diabetes_experiment")

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    # The predicted column is "progression" which is a quantitative measure of
    # disease progression one year after baseline
    train_x = train.drop(["progression"], axis=1)
    test_x = test.drop(["progression"], axis=1)
    train_y = train[["progression"]]
    test_y = test[["progression"]]

    if float(in_alpha) is None:
        alpha = 0.05
    else:
        alpha = float(in_alpha)

    if float(in_l1_ratio) is None:
        l1_ratio = 0.05
    else:
        l1_ratio = float(in_l1_ratio)

    # Start an MLflow run
    with mlflow.start_run():
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        predicted_qualities = lr.predict(test_x)

        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        # Print out ElasticNet model metrics
        print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        # Set tracking_URI first and then reset it back to not specifying port
        # Note, we had specified this in an earlier cell
        # mlflow.set_tracking_uri(mlflow_tracking_URI)

        # Log mlflow attributes for mlflow UI
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)
        mlflow.sklearn.log_model(lr, "model")
        modelpath = "models/model-%f-%f" % (
            alpha, l1_ratio)
        mlflow.sklearn.save_model(lr, modelpath)

        # Call plot_enet_descent_path
        plot_enet_descent_path(X, y, l1_ratio)

        # Log artifacts (output files)
        mlflow.log_artifact(
            "out/ElasticNet-paths_l1_ratio_{}_.png".format(str(l1_ratio)))


if __name__ == "__main__":

    # Load Diabetes datasets
    diabetes = datasets.load_diabetes()
    X = diabetes.data
    y = diabetes.target

    # Create pandas DataFrame for sklearn ElasticNet linear_model
    Y = np.array([y]).transpose()
    d = np.concatenate((X, Y), axis=1)
    cols = ["age", "sex", "bmi", "bp", "s1", "s2",
            "s3", "s4", "s5", "s6", "progression"]
    data = pd.DataFrame(d, columns=cols)

    if os.path.exists("models"):
        shutil.rmtree("models")

    os.mkdir("models")

    if os.path.exists("out"):
        set
        shutil.rmtree("out")

    os.mkdir("out")

    train_diabetes(data, 0.01, 0.01)
    train_diabetes(data, 0.01, 0.75)
    train_diabetes(data, 0.01, .5)
    train_diabetes(data, 0.01, 1)
