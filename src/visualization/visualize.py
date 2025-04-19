import pathlib
import joblib
import sys
import yaml
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import tree
from dvclive import Live
import matplotlib.pyplot as plt


def evaluate(model, X, y, split, live, save_path):
    """
    Dump all evaluation metrics and plots for given datasets.

    Args:
        model (sklearn.ensemble.RandomForestClassifier): Trained classifier.
        X (pandas.DataFrame): Input DF.
        y (pamdas.Series): Target column.
        split (str): Dataset name.
        live (dvclive.Live): Dvclive instance.
        save_path (str): Path to save the metrics.
    """
    prediction = model.predict(X)
    r2_score = metrics.r2_score(y, prediction)
    mse = metrics.mean_squared_error(y, prediction)
    rmse = mse ** 0.5

    if not live.summary:
        live.summary = {"RMSE": {}, "R2": {}}
    
    live.summary['RMSE'][split] = rmse
    live.summary['R2'][split] = r2_score

def save_importance_plot(live, model, feature_names):
    """
    Save feature importance plot.

    Args:
        live (dvclive.Live): Dvclive instance.
        model (sklearn.ensemble.RandomForestClassifier): Trained classifier.
        feature_names (list): List of feature names.
    """
    fig, axes = plt.subplots(dpi=100)
    fig.subplots_adjust(bottom=0.2, top=0.95)
    axes.set_ylabel("Importance")
    axes.set_title("Feature Importance")
    axes.bar(feature_names, model.feature_importances_)

    importance = model.feature_importances_
    forest_importances = pd.Series(importance, index=feature_names)
    std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0).nlargest(n=10)
    forest_importances.plot.bar(yerr=std, ax=axes)
    axes.set_title("Feature importances using MDI")
    axes.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()

    live.log_image("importance.png", fig)


def main():

    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent

    model_file = sys.argv[1]
    # Load the model.
    model = joblib.load(model_file)
    
    # Load the data.
    input_file = sys.argv[2]
    data_path = home_dir.as_posix() + input_file
    output_path = home_dir.as_posix() + '/dvclive'
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)

    TARGET = 'trip_duration'
    train_features = pd.read_csv(data_path + '/train.csv')
    X_train = train_features.drop(TARGET, axis=1)
    y_train = train_features[TARGET]
    feature_names = X_train.columns.to_list()

    test_features = pd.read_csv(data_path + '/test.csv')
    X_test = test_features.drop(TARGET, axis=1)
    y_test = test_features[TARGET]

    # Evaluate train and test datasets.
    with Live(output_path, dvcyaml=False) as live:
        evaluate(model, X_train, y_train, "train", live, output_path)
        evaluate(model, X_test, y_test, "test", live, output_path)

        # Dump feature importance plot.
        save_importance_plot(live, model, feature_names)

if __name__ == "__main__":
    main()