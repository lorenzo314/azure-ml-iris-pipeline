"""Creates (or updates) the Azure ML compute cluster with Managed Identity.

Run once. Managed Identity must be enabled to allow model registration
from within pipeline steps. Note the Principal ID shown in the Azure portal
under the cluster's Managed Identity tab — you will need it for assign_identity.
"""

from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import AmlCompute, IdentityConfiguration

from src.utils.config import load_config, get_azure_credentials

cfg = load_config()
subscription_id, resource_group, workspace_name = get_azure_credentials()

ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id,
    resource_group,
    workspace_name,
)

aml_cfg = cfg["azure_ml"]
compute = AmlCompute(
    name=aml_cfg["compute_name"],
    size=aml_cfg["compute_size"],
    min_instances=aml_cfg["compute_min_instances"],
    max_instances=aml_cfg["compute_max_instances"],
    identity=IdentityConfiguration(type="system_assigned"),
)

created_compute = ml_client.begin_create_or_update(compute).result()
print(f"Created compute: {created_compute.name}")
print("Managed Identity enabled.")
