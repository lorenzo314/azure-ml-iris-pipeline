import json
import joblib
import os
import glob
import traceback


def init():
    global model
    try:
        model_dir = os.getenv("AZUREML_MODEL_DIR")
        print("MODEL DIR:", model_dir)

        files = list(glob.glob(model_dir + "/**", recursive=True))
        print("FILES:", files)

        # find any .pkl file
        model_files = glob.glob(os.path.join(model_dir, "**/*.pkl"), recursive=True)

        if not model_files:
            raise Exception("No .pkl model file found")

        print("MODEL FILES:", model_files)

        model_path = model_files[0]
        print("LOADING MODEL:", model_path)

        model = joblib.load(model_path)

    except Exception as e:
        print("INIT ERROR:")
        traceback.print_exc()
        raise e


def run(raw_data):
    try:
        data = json.loads(raw_data)
        X = data["data"]
        preds = model.predict(X)
        return preds.tolist()

    except Exception as e:
        print("RUN ERROR:")
        traceback.print_exc()
        return str(e)
