# ARCHITECTURE.md: G-Force (GCP T4 SD1.5 Optimizer)

## 1. Project Mission & Vision
**G-Force** is a "Disposable Compute" orchestration engine. It treats high-end GPUs as ephemeral utilities—spun up for a specific task and destroyed immediately after—to ensure professional-grade ML workflows (training and inference) remain sustainable on a hobbyist budget (~$10/month).

### Strategic Goals:
* **Zero-Idle Billing:** Eliminate "forgotten VMs" by using Google Cloud Batch, which terminates and deletes instances automatically.
* **Cost-Safety First:** Prioritize 70-90% savings via Spot instances over boot speed.
* **Developer Ergonomics:** A "Single Command" CLI experience for complex ML-Ops tasks.
* **Reproducible Environments:** Standardized containerized workloads for consistent model outputs.

## 2. Core Use Cases
* **Subject Injection (DreamBooth):** Training SD 1.5 to recognize specific subjects (people, custom hardware like synthesizers, or art styles) for < $0.10 USD per run.
* **Batch Inference:** Generating thousands of images at scale (~3,000 images per $1.00 USD) for agentic workflows.
* **Dataset Maintenance:** Using on-demand VLM (Vision-Language Models) like Florence-2 to auto-caption large datasets sitting in GCS.

## 3. Technical Stack
* **Environment:** Python 3.12+ managed by **uv**.
* **Infrastructure (IaC):** **pulumi-gcp** (Python) via Automation API.
* **Orchestration:** **google-cloud-batch** (Python Client).
* **CLI:** **Typer** + **Rich**.
* **ML Backend:** **diffusers**, **accelerate**, **xformers**, **bitsandbytes**.

## 4. Mandatory Guardrails (Non-Negotiable)
The coding agent **must** implement these hard constraints:
1.  **Spot Enforcement:** All VM configurations must be hardcoded to `provisioning_model = "SPOT"`.
2.  **Hard Timeout:** Every Task Spec must include a `max_run_duration` of `3600s` (1 hour).
3.  **No Persistent Nodes:** Do not use `gce` instance resources; use `gcp.cloudbatch.Job`.
4.  **Confirm Before Burn:** CLI must display a **Rich** panel with estimated costs (~$0.15/hr) and require `[y/N]` confirmation.

## 5. Directory Structure
\```text
gforce-project/
├── pyproject.toml         # Managed by uv
├── Pulumi.yaml            # Pulumi project metadata
├── gforce/                
│   ├── cli/               
│   │   └── main.py        # Commands: infra-up, train, gen, status
│   ├── infra/             
│   │   └── stack.py       # Pulumi definitions (GCS, VPC, IAM)
│   ├── core/              
│   │   ├── batch.py       # google-cloud-batch Job construction
│   │   └── guardrails.py  # Cost estimation and Spot validation
│   └── scripts/           
│       └── worker_init.py # Script executed INSIDE the remote T4 VM
└── ARCHITECTURE.md        # This document
\```

## 6. Training Specifications (SD 1.5)
The code in **gforce/scripts/worker_init.py** must be optimized for the NVIDIA T4 (16GB VRAM):
* **Precision:** Always use `fp16` with `xformers` attention.
* **Optimizer:** Use `8-bit Adam` (bitsandbytes) to keep VRAM < 12GB.
* **Gradient Accumulation:** Hardcode to a minimum of 4 to ensure stability on the T4.
* **Checkpoints:** Write `.safetensors` directly to the `/mnt/disks/gcs` mount point.
* **Auto-Drivers:** Ensure the Batch Job spec has `install_gpu_drivers: true` to avoid manual driver management.

## 7. Success Metric
A user runs **uv run gforce train --dataset gs://my-bucket/dataset**, confirms the $0.15/hr warning, and a T4 GPU completes a DreamBooth run and self-destructs within 40 minutes for a total cost under $0.10.

## 8. Strategic Decisions & Defaults
* **State Backend:** Pulumi state is stored in the project GCS bucket (`gs://<bucket>/pulumi-state`).
* **Filesystem:** The worker VM utilizes **GCS Fuse** for POSIX-compliant access to datasets and model checkpoints.
* **Auth:** Leverages Google Application Default Credentials (ADC).
* **Model Loading:** Defaults to https://huggingface.co/TheImposterImposters/URPM-SD1.5-v2.3.inpainting; GCS-based caching is mandatory to reduce egress/ingress costs on repeated runs.
* **CLI UX:** Includes a `gforce pull` command to simplify artifact retrieval without requiring manual `gsutil` calls.

## 9. Initialization & Configuration Logic

### A. Bootstrap Flow (gforce init)
The CLI must implement a bootstrap sequence using the Google Python SDKs BEFORE invoking Pulumi:
1. **API Enablement:** Enable `batch.googleapis.com`, `compute.googleapis.com`, and `storage.googleapis.com`.
2. **State Bucket:** Create `gs://gforce-assets-<project_id>` with Uniform Bucket-Level Access.
3. **Pulumi Login:** Execute `pulumi login gs://gforce-assets-<project_id>/state`.

### B. Intelligent Caching
* **Storage Path:** `gs://<bucket>/cache/models/{provider}/{repo_id}/{commit_hash}`.
* **Worker Logic:** The T4 worker must check for existing cache manifests. If found, it performs a local sync to the VM's high-speed boot disk (50GB) to minimize training latency.

### C. Resource Lifecycle Guardrails
* **Job TTL:** All Cloud Batch tasks MUST have `max_run_duration` set to `3600s`.
* **Cost Disclosure:** CLI must fetch current regional Spot prices (via Cloud Billing API or hardcoded estimate) and prompt: "Estimated cost for this 1hr run: $0.14. Proceed? [y/N]".

### D. Authentication
* **Primary:** Default to Application Default Credentials (ADC).
* **Validation:** On every command, `gforce` verifies credentials. If invalid, it prints: `Please run: gcloud auth application-default login`.
Next Step: The Code Foundation
