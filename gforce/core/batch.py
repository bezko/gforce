"""Cloud Batch job construction for G-Force."""

import logging
from dataclasses import dataclass
from typing import Any

from google.cloud import batch_v1
from google.protobuf import duration_pb2

from gforce.core.config import GForceConfig, get_config
from gforce.core.guardrails import enforce_guardrails

logger = logging.getLogger(__name__)


@dataclass
class JobConfig:
    """Configuration for a Batch job."""

    job_name: str
    container_image: str
    command: list[str]
    machine_type: str = "n1-standard-4"
    gpu_type: str = "nvidia-tesla-t4"
    gpu_count: int = 1
    disk_size_gb: int = 50
    max_duration_seconds: int = 3600
    provisioning_model: str = "SPOT"
    gcs_bucket: str | None = None
    environment_variables: dict[str, str] | None = None
    service_account: str | None = None
    use_custom_worker_image: bool = False


class BatchJobBuilder:
    """Builds Cloud Batch job specifications."""

    DEFAULT_CONTAINER_IMAGE = "python:3.12-slim"
    CUSTOM_WORKER_IMAGE = "worker"

    def __init__(self, config: GForceConfig | None = None):
        self.config = config or get_config()

    def _create_runnable(
        self,
        job_config: JobConfig,
    ) -> batch_v1.Runnable:
        """Create a runnable (container) specification."""
        env_vars = job_config.environment_variables or {}

        # Add default env vars
        env_vars.setdefault("PYTHONUNBUFFERED", "1")
        env_vars.setdefault("HF_HOME", "/tmp/huggingface")
        env_vars.setdefault("TRANSFORMERS_CACHE", "/tmp/huggingface")

        container = batch_v1.Runnable.Container(
            image_uri=job_config.container_image or self.DEFAULT_CONTAINER_IMAGE,
            commands=job_config.command,
            options="--privileged",  # Required for GCS Fuse
        )

        return batch_v1.Runnable(
            container=container,
            environment=batch_v1.Environment(
                variables=env_vars,
            ),
        )

    def _create_task_spec(
        self,
        job_config: JobConfig,
    ) -> batch_v1.TaskSpec:
        """Create a task specification."""
        runnable = self._create_runnable(job_config)

        # Compute resources
        compute_resource = batch_v1.ComputeResource(
            cpu_milli=4000,  # 4 vCPUs
            memory_mib=15360,  # 15 GB
        )

        # Max run duration
        max_duration = duration_pb2.Duration(seconds=job_config.max_duration_seconds)

        # GCS volume mount via GCS Fuse
        gcs_volume = batch_v1.Volume(
            gcs=batch_v1.GCS(
                remote_path=job_config.gcs_bucket or self.config.get_bucket_name(),
            ),
            mount_path="/mnt/disks/gcs",
            mount_options=[
                "implicit-dirs",
                "file-mode=777",
                "dir-mode=777",
            ],
        )

        return batch_v1.TaskSpec(
            runnables=[runnable],
            compute_resource=compute_resource,
            max_run_duration=max_duration,
            volumes=[gcs_volume],
            environment=batch_v1.Environment(
                variables={
                    "GPU_TYPE": job_config.gpu_type,
                    "GCS_BUCKET": job_config.gcs_bucket or self.config.get_bucket_name(),
                }
            ),
        )

    def _create_allocation_policy(
        self,
        job_config: JobConfig,
    ) -> batch_v1.AllocationPolicy:
        """Create an allocation policy with Spot enforcement."""
        # Instance policy with GPU and Spot provisioning
        instance_policy = batch_v1.AllocationPolicy.InstancePolicy(
            machine_type=job_config.machine_type,
            accelerators=[
                batch_v1.AllocationPolicy.Accelerator(
                    type_=job_config.gpu_type,
                    count=job_config.gpu_count,
                )
            ],
            boot_disk=batch_v1.AllocationPolicy.Disk(
                type_="pd-ssd",
                size_gb=job_config.disk_size_gb,
            ),
            provisioning_model=batch_v1.AllocationPolicy.ProvisioningModel.SPOT,
        )

        # Instance policy or template
        instances = batch_v1.AllocationPolicy.InstancePolicyOrTemplate(
            policy=instance_policy,
            install_gpu_drivers=True,
        )

        # Location policy (any available zone)
        location_policy = batch_v1.AllocationPolicy.LocationPolicy(
            allowed_locations=[
                f"zones/{self.config.gcp_zone}",
            ],
        )

        # Network policy (default VPC)
        network_policy = batch_v1.AllocationPolicy.NetworkPolicy(
            network_interfaces=[
                batch_v1.AllocationPolicy.NetworkInterface(
                    network="global/networks/default",
                )
            ]
        )

        # Service account
        service_account = None
        if job_config.service_account:
            service_account = batch_v1.ServiceAccount(
                email=job_config.service_account,
            )

        return batch_v1.AllocationPolicy(
            instances=[instances],
            location=location_policy,
            network=network_policy,
            service_account=service_account,
        )

    def build_job(self, job_config: JobConfig) -> batch_v1.Job:
        """Build a complete Batch job specification.

        Args:
            job_config: Job configuration

        Returns:
            Complete Batch job specification
        """
        task_spec = self._create_task_spec(job_config)

        task_group = batch_v1.TaskGroup(
            task_spec=task_spec,
            task_count=1,
            parallelism=1,
        )

        allocation_policy = self._create_allocation_policy(job_config)

        job = batch_v1.Job(
            name=job_config.job_name,
            task_groups=[task_group],
            allocation_policy=allocation_policy,
            logs_policy=batch_v1.LogsPolicy(
                destination=batch_v1.LogsPolicy.Destination.CLOUD_LOGGING,
            ),
        )

        # Validate guardrails
        job_dict = self._job_to_dict(job)
        enforce_guardrails(job_dict)

        return job

    def _job_to_dict(self, job: batch_v1.Job) -> dict[str, Any]:
        """Convert a Job to a dictionary for validation."""
        # Convert to dict for guardrail validation
        result: dict[str, Any] = {
            "taskGroups": [],
            "allocationPolicy": {},
        }
        
        # Extract task groups
        for tg in job.task_groups:
            task_spec_dict: dict[str, Any] = {}
            if tg.task_spec.max_run_duration:
                task_spec_dict["maxRunDuration"] = f"{tg.task_spec.max_run_duration.seconds}s"
            result["taskGroups"].append({"taskSpec": task_spec_dict})
        
        # Extract allocation policy
        if job.allocation_policy and job.allocation_policy.instances:
            instances = []
            for inst in job.allocation_policy.instances:
                if inst.policy:
                    policy_dict: dict[str, Any] = {}
                    if inst.policy.provisioning_model:
                        policy_dict["provisioningModel"] = inst.policy.provisioning_model.name
                    instances.append({"policy": policy_dict})
            result["allocationPolicy"] = {"instances": instances}
        
        return result


class BatchJobClient:
    """Client for interacting with Cloud Batch."""

    def __init__(self, config: GForceConfig | None = None):
        self.config = config or get_config()
        self.client = batch_v1.BatchServiceClient()
        self.builder = BatchJobBuilder(self.config)

    def get_parent(self) -> str:
        """Get the parent resource path for jobs."""
        return f"projects/{self.config.gcp_project}/locations/{self.config.gcp_region}"

    def create_job(self, job_config: JobConfig) -> batch_v1.Job:
        """Create and submit a Batch job.

        Args:
            job_config: Job configuration

        Returns:
            Created job
        """
        job = self.builder.build_job(job_config)

        request = batch_v1.CreateJobRequest(
            parent=self.get_parent(),
            job=job,
            job_id=job_config.job_name,
        )

        logger.info(f"Creating Batch job: {job_config.job_name}")
        response = self.client.create_job(request=request)
        logger.info(f"Created job: {response.name}")

        return response

    def get_job(self, job_name: str) -> batch_v1.Job:
        """Get a job by name.

        Args:
            job_name: Job name (full path or just ID)

        Returns:
            Job
        """
        if "/" not in job_name:
            job_name = f"{self.get_parent()}/jobs/{job_name}"

        return self.client.get_job(batch_v1.GetJobRequest(name=job_name))

    def list_jobs(self, filter_str: str | None = None) -> list[batch_v1.Job]:
        """List jobs in the configured project/region.

        Args:
            filter_str: Optional filter string

        Returns:
            List of jobs
        """
        request = batch_v1.ListJobsRequest(
            parent=self.get_parent(),
            filter=filter_str,
        )

        return list(self.client.list_jobs(request=request))

    def delete_job(self, job_name: str) -> None:
        """Delete a job.

        Args:
            job_name: Job name (full path or just ID)
        """
        if "/" not in job_name:
            job_name = f"{self.get_parent()}/jobs/{job_name}"

        logger.info(f"Deleting job: {job_name}")
        self.client.delete_job(batch_v1.DeleteJobRequest(name=job_name))


def create_training_job(
    dataset_uri: str,
    output_name: str,
    model_id: str | None = None,
    instance_prompt: str | None = None,
    num_steps: int = 1000,
    hf_token: str | None = None,
    use_custom_image: bool = True,
    config: GForceConfig | None = None,
) -> batch_v1.Job:
    """Create a DreamBooth training job.

    Args:
        dataset_uri: GCS URI to the training dataset
        output_name: Name for the output checkpoint
        model_id: HuggingFace model ID (uses config default if None)
        instance_prompt: Prompt for the instance (e.g., "photo of sks person")
        num_steps: Number of training steps
        config: GForceConfig instance

    Returns:
        Created job
    """
    cfg = config or get_config()
    model = model_id or cfg.default_model

    job_name = f"gforce-train-{output_name}-{int(__import__('time').time())}"
    job_name = job_name.replace("_", "-").lower()[:63]  # GCP naming constraints

    # Parse dataset path
    bucket_name = dataset_uri.replace("gs://", "").split("/")[0]

    # Determine container image
    if use_custom_image:
        # Use pre-built worker image from Artifact Registry
        container_image = f"{cfg.gcp_region}-docker.pkg.dev/{cfg.gcp_project}/gforce/worker:latest"
        # Custom image has entrypoint, so command is just args
        command = [
            "--mode", "train",
            "--dataset", dataset_uri,
            "--model", model,
            "--output", output_name,
            "--steps", str(num_steps),
        ]
    else:
        # Use base Python image and run script manually
        container_image = "python:3.12-slim"
        command = [
            "python",
            "/mnt/disks/gcs/scripts/worker_init.py",
            "--mode", "train",
            "--dataset", dataset_uri,
            "--model", model,
            "--output", output_name,
            "--steps", str(num_steps),
        ]

    # Build environment variables
    env_vars = {
        "INSTANCE_PROMPT": instance_prompt or "",
        "MODEL_ID": model,
        "OUTPUT_NAME": output_name,
        "GCS_BUCKET": bucket_name,
    }
    
    # Add HF_TOKEN if provided (for gated models)
    if hf_token:
        env_vars["HF_TOKEN"] = hf_token

    job_config = JobConfig(
        job_name=job_name,
        container_image=container_image,
        command=command,
        machine_type=cfg.machine_type,
        gpu_type=cfg.gpu_type,
        gpu_count=cfg.gpu_count,
        max_duration_seconds=cfg.max_run_duration,
        gcs_bucket=bucket_name,
        environment_variables=env_vars,
        use_custom_worker_image=use_custom_image,
    )

    client = BatchJobClient(cfg)
    return client.create_job(job_config)


def create_inference_job(
    prompt: str,
    model_path: str | None = None,
    num_images: int = 10,
    output_prefix: str = "gen",
    hf_token: str | None = None,
    use_custom_image: bool = True,
    config: GForceConfig | None = None,
) -> batch_v1.Job:
    """Create a batch inference job.

    Args:
        prompt: Generation prompt
        model_path: Path to the model (uses cached if None)
        num_images: Number of images to generate
        output_prefix: Prefix for output files
        config: GForceConfig instance

    Returns:
        Created job
    """
    cfg = config or get_config()

    job_name = f"gforce-gen-{output_prefix}-{int(__import__('time').time())}"
    job_name = job_name.replace("_", "-").lower()[:63]

    # Determine container image
    if use_custom_image:
        container_image = f"{cfg.gcp_region}-docker.pkg.dev/{cfg.gcp_project}/gforce/worker:latest"
        command = [
            "--mode", "inference",
            "--prompt", prompt,
            "--num-images", str(num_images),
            "--output-prefix", output_prefix,
        ]
    else:
        container_image = "python:3.12-slim"
        command = [
            "python",
            "/mnt/disks/gcs/scripts/worker_init.py",
            "--mode", "inference",
            "--prompt", prompt,
            "--num-images", str(num_images),
            "--output-prefix", output_prefix,
        ]

    # Build environment variables
    env_vars = {
        "GENERATION_PROMPT": prompt,
        "MODEL_PATH": model_path or "",
        "NUM_IMAGES": str(num_images),
        "GCS_BUCKET": cfg.get_bucket_name(),
    }
    
    if hf_token:
        env_vars["HF_TOKEN"] = hf_token

    job_config = JobConfig(
        job_name=job_name,
        container_image=container_image,
        command=command,
        machine_type=cfg.machine_type,
        gpu_type=cfg.gpu_type,
        gpu_count=cfg.gpu_count,
        max_duration_seconds=cfg.max_run_duration,
        gcs_bucket=cfg.get_bucket_name(),
        environment_variables=env_vars,
        use_custom_worker_image=use_custom_image,
    )

    client = BatchJobClient(cfg)
    return client.create_job(job_config)


def get_job_status(job_name: str, config: GForceConfig | None = None) -> dict:
    """Get the status of a job.

    Args:
        job_name: Job name
        config: GForceConfig instance

    Returns:
        Dictionary with job status information
    """
    cfg = config or get_config()
    client = BatchJobClient(cfg)
    job = client.get_job(job_name)

    status = job.status
    return {
        "name": job.name,
        "state": status.state.name if status else "UNKNOWN",
        "uid": job.uid,
        "create_time": job.create_time.isoformat() if job.create_time else None,
        "update_time": job.update_time.isoformat() if job.update_time else None,
    }


def list_active_jobs(config: GForceConfig | None = None) -> list[dict]:
    """List all active (non-terminated) jobs.

    Args:
        config: GForceConfig instance

    Returns:
        List of job status dictionaries
    """
    cfg = config or get_config()
    client = BatchJobClient(cfg)

    jobs = client.list_jobs()
    active_states = {
        "STATE_UNSPECIFIED",
        "QUEUED",
        "SCHEDULED",
        "RUNNING",
    }

    return [
        {
            "name": job.name,
            "state": job.status.state.name if job.status else "UNKNOWN",
            "create_time": job.create_time.isoformat() if job.create_time else None,
        }
        for job in jobs
        if job.status and job.status.state.name in active_states
    ]
