"""Creates (or updates) the Azure ML environment.

Run once before the first pipeline execution, or when conda.yaml changes.
"""

from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Environment

from src.utils.config import load_config, get_azure_credentials

cfg = load_config()
subscription_id, resource_group, workspace_name = get_azure_credentials()

ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id,
    resource_group,
    workspace_name,
)

env = Environment(
    name=cfg["azure_ml"]["environment_name"],
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
    conda_file="conda.yaml",
)

created_env = ml_client.environments.create_or_update(env)
print(f"Created environment: {created_env.name} version {created_env.version}")
