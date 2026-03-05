# G-Force: Disposable Compute for ML Workflows

G-Force is a "Disposable Compute" orchestration engine for Google Cloud Platform. It treats high-end GPUs as ephemeral utilities—spun up for a specific ML task and destroyed immediately after—to ensure professional-grade ML workflows remain sustainable on a hobbyist budget (~$10/month).

## Features

- **Zero-Idle Billing**: Uses Google Cloud Batch with automatic termination
- **Spot Instances**: 70-90% cost savings through Spot (preemptible) instances
- **One-Command Experience**: Simple CLI for complex ML-Ops tasks
- **Reproducible Environments**: Containerized workloads for consistent outputs
- **Model Caching**: GCS-based model caching to reduce egress costs

## Supported Workflows

1. **DreamBooth Training**: Fine-tune SD 1.5 to recognize specific subjects for < $0.10 per run
2. **Batch Inference**: Generate thousands of images at scale (~3,000 images per $1.00)
3. **Dataset Maintenance**: Auto-caption datasets using Vision-Language Models

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd gforce

# Install with uv
uv pip install -e ".[dev]"

# Or with pip
pip install -e ".[dev]"
```

## Quick Start

### 1. Authenticate with GCP

```bash
gcloud auth application-default login
```

### 2. Initialize G-Force

```bash
uv run gforce init
```

This will:
- Enable required GCP APIs (Cloud Batch, Compute Engine, Cloud Storage)
- Create the GCS bucket for assets (`gforce-assets-<project_id>`)
- Configure Pulumi state backend
- Deploy base infrastructure (VPC, Service Accounts, IAM)

### 3. Run Training

```bash
# Upload your dataset to GCS first
gsutil -m cp -r ./my-dataset gs://gforce-assets-<project_id>/datasets/

# Run DreamBooth training
uv run gforce train gs://gforce-assets-<project_id>/datasets/my-dataset \
  --output my-model \
  --prompt "photo of sks person"
```

### 4. Check Status

```bash
uv run gforce status
```

### 5. Download Results

```bash
uv run gforce pull my-model ./outputs
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `gforce init` | Initialize G-Force infrastructure |
| `gforce infra-up` | Deploy/update infrastructure |
| `gforce infra-down` | Destroy infrastructure |
| `gforce train <dataset>` | Run DreamBooth training |
| `gforce gen <prompt>` | Run batch inference |
| `gforce status [job-id]` | Check job status |
| `gforce pull <output>` | Download outputs from GCS |
| `gforce cache-list` | List cached models |
| `gforce auth-status` | Check authentication |

## Configuration

Configuration can be provided via environment variables:

```bash
export GFORCE_GCP_PROJECT="my-project"
export GFORCE_GCP_REGION="us-central1"
export GFORCE_SPOT_PRICE_ESTIMATE="0.15"
export GFORCE_MAX_RUN_DURATION="3600"
```

Or use command-line flags:

```bash
uv run gforce --project my-project --region europe-west1 train ...
```

## Architecture

```
gforce/
├── cli/
│   └── main.py          # CLI commands
├── core/
│   ├── auth.py          # ADC validation
│   ├── batch.py         # Cloud Batch job construction
│   ├── bootstrap.py     # Initialization logic
│   ├── cache.py         # Model caching
│   ├── config.py        # Configuration management
│   └── guardrails.py    # Cost safety & validation
├── infra/
│   └── stack.py         # Pulumi infrastructure definitions
└── scripts/
    └── worker_init.py   # T4 VM training script
```

## Safety Guardrails

G-Force enforces strict safety measures:

1. **Spot Enforcement**: All VMs must use Spot provisioning (70-90% savings)
2. **Hard Timeout**: All jobs auto-terminate after 1 hour maximum
3. **Cost Confirmation**: CLI displays estimated costs and requires confirmation
4. **No Persistent Nodes**: Only uses Cloud Batch, never GCE instances

## Testing

Run the test suite:

```bash
# All tests
uv run pytest

# Unit tests only
uv run pytest tests/unit

# Integration tests
uv run pytest tests/integration -m integration

# With coverage
uv run pytest --cov=gforce --cov-report=html
```

## Cost Estimates

| Workflow | Estimated Cost |
|----------|---------------|
| DreamBooth Training (1000 steps, ~40 min) | ~$0.10 |
| Batch Inference (100 images) | ~$0.03 |
| Dataset Captioning (1000 images) | ~$0.05 |

*Based on Spot T4 pricing at $0.15/hour*

## Requirements

- Python 3.12+
- GCP project with billing enabled
- `gcloud` CLI configured
- Pulumi CLI (optional, for infrastructure management)

## License

MIT License - See LICENSE file for details.
