import argparse
import os
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("--model_dirs", type=str, nargs="+")
parser.add_argument("--best_model_dir", type=str)

args = parser.parse_args()

best_acc = -1
best_model_path = None

for d in args.model_dirs:
    metrics_path = os.path.join(d, "metrics.txt")

    if not os.path.exists(metrics_path):
        continue

    with open(metrics_path, "r") as f:
        acc = float(f.read())

    print(f"{d} -> accuracy: {acc}")

    if acc > best_acc:
        best_acc = acc
        best_model_path = os.path.join(d, "model.pkl")

# Copy best model
os.makedirs(args.best_model_dir, exist_ok=True)

shutil.copy(best_model_path, os.path.join(args.best_model_dir, "model.pkl"))

print("Best accuracy:", best_acc)
