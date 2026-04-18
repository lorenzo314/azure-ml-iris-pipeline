"""Defines and submits the Iris training pipeline.

The pipeline runs three independent training jobs, selects the best model
by accuracy, and registers it in the Azure ML model registry.
"""

from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient, command, dsl, Input, Output

from src.utils.config import load_config, get_azure_credentials

cfg = load_config()
subscription_id, resource_group, workspace_name = get_azure_credentials()

aml_cfg = cfg["azure_ml"]
environment = f"{aml_cfg['environment_name']}:{aml_cfg['environment_version']}"
compute = aml_cfg["compute_name"]

ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id,
    resource_group,
    workspace_name,
)

# --- Pipeline components -------------------------------------------------

train_component = command(
    name="train_iris",
    code="./src/",
    command="python train.py --output_dir ${{outputs.model_output}}",
    environment=environment,
    compute=compute,
    outputs={"model_output": Output(type="uri_folder")},
)

select_component = command(
    name="select_best",
    code="./src/",
    command=(
        "python select_best.py "
        "--model_dirs ${{inputs.m1}} ${{inputs.m2}} ${{inputs.m3}} "
        "--best_model_dir ${{outputs.best_model}}"
    ),
    environment=environment,
    compute=compute,
    inputs={
        "m1": Input(type="uri_folder"),
        "m2": Input(type="uri_folder"),
        "m3": Input(type="uri_folder"),
    },
    outputs={"best_model": Output(type="uri_folder")},
)

register_component = command(
    name="register_model",
    code="./",
    command="python src/register_model.py --model_dir ${{inputs.model_dir}}",
    environment=environment,
    compute=compute,
    inputs={"model_dir": Input(type="uri_folder")},
    # Pass workspace credentials so the step can call the ML client
    environment_variables={
        "SUBSCRIPTION_ID": subscription_id,
        "RESOURCE_GROUP": resource_group,
        "WORKSPACE_NAME": workspace_name,
    },
)

# --- Pipeline definition -------------------------------------------------

@dsl.pipeline(
    compute=compute,
    description="Iris training pipeline: train x3 → select best → register",
)
def iris_pipeline():
    t1 = train_component()
    t2 = train_component()
    t3 = train_component()

    select = select_component(
        m1=t1.outputs.model_output,
        m2=t2.outputs.model_output,
        m3=t3.outputs.model_output,
    )

    register_component(model_dir=select.outputs.best_model)

    return {}


# --- Submit --------------------------------------------------------------

pipeline_job = ml_client.jobs.create_or_update(
    iris_pipeline(),
    experiment_name=aml_cfg["experiment_name"],
)
print(f"Pipeline submitted: {pipeline_job.name}")

