"""Deploys a registered model to an Azure ML Managed Online Endpoint.

Model name, version, and deployment settings are read from configs/config.yaml.
Workspace credentials are passed as CLI arguments (populated by
scripts/submit_endpoint_creation_job from environment variables).
"""

from __future__ import annotations

import argparse
import uuid
from pathlib import Path

from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    CodeConfiguration,
    Environment,
    ManagedOnlineDeployment,
    ManagedOnlineEndpoint,
)
from azure.identity import DefaultAzureCredential

from src.utils.config import load_config

# Repo root is one level up from this file (deploy/deploy.py)
_REPO_ROOT = Path(__file__).resolve().parent.parent


def deploy_model(
    subscription_id: str,
    resource_group: str,
    workspace_name: str,
    endpoint_name: str,
) -> None:
    """Deploy the registered model to a Managed Online Endpoint."""
    cfg = load_config()
    model_cfg = cfg["model"]
    deploy_cfg = cfg["deployment"]

    credential = DefaultAzureCredential()
    ml_client = MLClient(
        credential=credential,
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace_name,
    )

    print("Creating / updating endpoint...")
    endpoint = ManagedOnlineEndpoint(name=endpoint_name, auth_mode="key")
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()

    print("Deploying model...")
    deployment = ManagedOnlineDeployment(
        name=deploy_cfg["deployment_name"],
        endpoint_name=endpoint_name,
        model=f"azureml:{model_cfg['name']}:{model_cfg['version']}",
        environment=Environment(
            image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
            conda_file=str(_REPO_ROOT / "conda_inference.yaml"),
        ),
        code_configuration=CodeConfiguration(
            code=str(_REPO_ROOT / "src"),
            scoring_script="score.py",
        ),
        instance_type=deploy_cfg["instance_type"],
        instance_count=deploy_cfg["instance_count"],
    )
    ml_client.online_deployments.begin_create_or_update(deployment).result()

    print("Routing 100% of traffic to deployment...")
    endpoint.traffic = {deploy_cfg["deployment_name"]: 100}
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()

    print(f"Deployment complete. Endpoint: {endpoint_name}")


if __name__ == "__main__":
    default_endpoint_name = f"iris-endpoint-{uuid.uuid4().hex[:8]}"

    parser = argparse.ArgumentParser()
    parser.add_argument("--subscription_id", required=True)
    parser.add_argument("--resource_group", required=True)
    parser.add_argument("--workspace_name", required=True)
    parser.add_argument("--endpoint_name", default=default_endpoint_name)
    args = parser.parse_args()

    deploy_model(
        subscription_id=args.subscription_id,
        resource_group=args.resource_group,
        workspace_name=args.workspace_name,
        endpoint_name=args.endpoint_name,
    )

