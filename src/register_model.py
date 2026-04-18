"""Registers the best model in the Azure ML model registry.

Called as a pipeline step by pipeline.py — not intended to be run directly.
Workspace credentials are injected as environment variables by the pipeline.
"""

import argparse
import os

from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model

from utils.config import load_config, get_azure_credentials

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, required=True)
args = parser.parse_args()

cfg = load_config()
subscription_id, resource_group, workspace_name = get_azure_credentials()

ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id,
    resource_group,
    workspace_name,
)

model_name = cfg["model"]["name"]
print(f"Registering model '{model_name}' from: {args.model_dir}")

model = Model(
    path=args.model_dir,
    name=model_name,
    type="custom_model",
)
ml_client.models.create_or_update(model)
print("Model registered successfully.")
