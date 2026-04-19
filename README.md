# azure-ml-iris-pipeline

![Python](https://img.shields.io/badge/python-3.11-blue)
![Azure ML](https://img.shields.io/badge/azure--ai--ml-1.32.0-0078D4?logo=microsoftazure)
![License](https://img.shields.io/badge/license-MIT-green)
![CI](https://github.com/lorenzo314/azure-ml-iris-pipeline/actions/workflows/ci.yaml/badge.svg)

An end-to-end MLOps pipeline built on Azure ML — from training to a live REST endpoint.
This is an intentionally simple project (Iris dataset, Random Forest) designed to demonstrate
a professional Azure ML workflow that can serve as a template for more complex work.

---

## Overview

The pipeline trains three Random Forest models on the Iris dataset, selects the best by
accuracy, registers it in the Azure ML model registry, and deploys it to a Managed Online
Endpoint — all orchestrated through Azure ML's pipeline SDK.

The focus here is on **workflow structure and MLOps practices** rather than model complexity:
environment and compute management, pipeline orchestration, model registration, managed
identity, and endpoint deployment are all production-grade patterns regardless of the dataset.

---

## Architecture

```
setup_environment.py       → Create Azure ML environment (once)
setup_compute.py           → Provision compute cluster + Managed Identity (once)
        │
        ▼
pipeline.py
  ├── train.py  ──┐
  ├── train.py  ──┼──▶  select_best.py  ──▶  register_model.py
  └── train.py  ──┘
        │
        ▼
deploy/deploy.py           → Create Managed Online Endpoint + deploy model
deploy/test_endpoint.py    → Retrieve scoring URI and API key
scripts/test_api           → Smoke-test the live endpoint via curl
```

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/lorenzo314/azure-ml-iris-pipeline.git
cd azure-ml-iris-pipeline
pip install -r requirements.txt
pip install -e .   # installs src/ as a package — required for all scripts to work
```

### 2. Configure credentials

```bash
cp .env.example .env
# Edit .env and fill in your Azure values
```

Your `.env` should look like this (never commit this file):

```bash
SUBSCRIPTION_ID=your-subscription-id
RESOURCE_GROUP=your-resource-group
WORKSPACE_NAME=your-workspace-name
COMPUTE_PRINCIPAL_ID=your-compute-principal-id
VERIFY_ASSIGNEE_ID=$COMPUTE_PRINCIPAL_ID   # same value as COMPUTE_PRINCIPAL_ID
```

> **Note:** shell scripts load `.env` automatically — no need to source it manually
> before running them. For Python scripts, your `.env` variables must be present
> in your shell session. You can load them once with:
> ```bash
> set -a && source .env && set +a
> ```

### 3. Make scripts executable (once after cloning)

```bash
chmod +x scripts/assign_identity scripts/verify_identity scripts/submit_endpoint_creation_job scripts/test_api
```

### 4. One-time setup (skip if environment and compute already exist)

```bash
python -m setup.setup_environment
python -m setup.setup_compute
./scripts/assign_identity
```

> **Note:** the `-m` flag is required when running from the repo root because
> `setup/` is a Python package. Running `python setup/setup_environment.py`
> directly will cause import errors.

### 5. Run the pipeline

```bash
python pipeline.py
```

Monitor progress in the Azure ML Studio UI. Once complete, the best model will be
registered in the model registry.

### 6. Deploy

Update `model.version` in `configs/config.yaml` to match the registered version, then:

```bash
./scripts/submit_endpoint_creation_job
```

### 7. Test the endpoint

```bash
python deploy/test_endpoint.py
# Copy the printed URI and key into your .env as ENDPOINT_URI and ENDPOINT_API_KEY
./scripts/test_api
```

---

## Project Structure

```
azure-ml-iris-pipeline/
│
├── .github/workflows/
│   └── ci.yaml                 # Lint and test on every push
│
├── configs/
│   └── config.yaml             # All non-sensitive configuration
│
├── deploy/
│   ├── deploy.py               # Create endpoint and deploy model
│   └── test_endpoint.py        # Retrieve scoring URI and API key
│
├── scripts/
│   ├── assign_identity         # Assign Contributor role to compute identity
│   ├── verify_identity         # Verify role assignment
│   ├── submit_endpoint_creation_job  # Wrapper to call deploy.py
│   └── test_api                # curl smoke-test against live endpoint
│
├── setup/
│   ├── setup_environment.py    # Create Azure ML environment
│   └── setup_compute.py        # Provision compute cluster
│
├── src/
│   ├── train.py                # Training script (called 3x by pipeline)
│   ├── select_best.py          # Selects best model by accuracy
│   ├── register_model.py       # Registers best model in Azure ML
│   ├── score.py                # Scoring script for the endpoint
│   └── utils/
│       └── config.py           # Shared config and credential loader
│
├── tests/
│   ├── test_auth.py            # Verify workspace connectivity (local only)
│   └── test_config.py          # Unit tests for config loader (CI)
│
├── pipeline.py                 # Main pipeline definition and submission
├── .env.example                # Environment variable template
├── requirements.txt            # Runtime dependencies
├── requirements-dev.txt        # Development and CI dependencies
├── pyproject.toml              # Package definition and tool config
├── conda.yaml                  # Azure ML training environment definition
└── conda_inference.yaml        # Azure ML inference environment definition
```

---

## Configuration

All non-sensitive settings live in `configs/config.yaml`:

```yaml
azure_ml:
  environment_name: mlflow-env
  environment_version: "7"
  compute_name: cpu-cluster
  ...

model:
  name: iris-rf-model
  version: "43"        # ← update this after each pipeline run

deployment:
  instance_type: Standard_F2s_v2
  ...
```

Sensitive values (`SUBSCRIPTION_ID`, `RESOURCE_GROUP`, `WORKSPACE_NAME`, principal IDs)
are always read from environment variables and never stored in the codebase.

---

## Tech Stack

| Component | Technology |
|---|---|
| Cloud platform | Microsoft Azure |
| ML orchestration | Azure ML Pipelines SDK v2 |
| Model training | scikit-learn 1.8 — Random Forest |
| Experiment tracking | MLflow 3.9 |
| Endpoint | Azure ML Managed Online Endpoint |
| Language | Python 3.11 |

---

## Limitations & Future Work

This project is intentionally kept simple to focus on pipeline structure:

- The three training runs are identical (same data, same algorithm) — future versions
  will use proper hyperparameter sweeps via Azure ML's `sweep()` component
- The dataset is the built-in Iris dataset — future versions will use registered
  Azure ML datasets with proper data versioning
- Tests cover connectivity only — future versions will include unit tests for
  training logic and data validation

---

## License

MIT — see [LICENSE](LICENSE)
