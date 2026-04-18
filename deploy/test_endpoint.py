"""Retrieves and prints the scoring URI and API key for the deployed endpoint.

Run after deploy.py. Copy the printed values into your .env file as
ENDPOINT_URI and ENDPOINT_API_KEY, then run scripts/test_api to verify
the live endpoint.
"""

from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient

from src.utils.config import load_config, get_azure_credentials

cfg = load_config()
subscription_id, resource_group, workspace_name = get_azure_credentials()

ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id,
    resource_group,
    workspace_name,
)

# List all online endpoints and print URI + key for each
endpoints = ml_client.online_endpoints.list()
for endpoint in endpoints:
    keys = ml_client.online_endpoints.get_keys(endpoint.name)
    print(f"Endpoint : {endpoint.name}")
    print(f"  Scoring URI : {endpoint.scoring_uri}")
    print(f"  API key     : {keys.primary_key}")
    print()
print("Add ENDPOINT_URI and ENDPOINT_API_KEY to your .env, then run scripts/test_api.")
