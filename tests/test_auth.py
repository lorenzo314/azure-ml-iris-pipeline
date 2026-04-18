"""Verifies that Azure ML workspace credentials are correctly configured.

Run this before anything else to confirm your environment variables are set
and that authentication works end-to-end.
"""

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

from src.utils.config import get_azure_credentials

subscription_id, resource_group, workspace_name = get_azure_credentials()

ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id,
    resource_group,
    workspace_name,
)

workspace = ml_client.workspaces.get(workspace_name)
print(f"Successfully connected to workspace: {workspace.name}")
