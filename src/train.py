import argparse
import os

import mlflow
# import azureml.mlflow
import mlflow.sklearn

from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", type=str)
args = parser.parse_args()

# Load data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model + search
model = RandomForestClassifier()
param_dist = {
    "n_estimators": [10, 50, 100],
    "max_depth": [3, 5, 10]
}
search = RandomizedSearchCV(model, param_dist, n_iter=3)

# Train
search.fit(X_train, y_train)
best_model = search.best_estimator_
acc = search.score(X_test, y_test)

# MLflow logging (run managed by Azure ML)
mlflow.log_param("search_type", "random")
mlflow.log_params(search.best_params_)
mlflow.log_metric("accuracy", acc)

# Save to pipeline output folder (for pipeline.py registration)
mlflow.sklearn.save_model(best_model, args.output_dir)

# Also log as MLflow artifact (for register_best.py)
mlflow.sklearn.log_model(best_model, "model")

# Save accuracy to file for pipeline
with open(os.path.join(args.output_dir, "metrics.txt"), "w") as f:
    f.write(str(acc))

