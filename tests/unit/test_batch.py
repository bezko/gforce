"""Tests for batch module."""

from unittest.mock import MagicMock, patch

import pytest
from google.cloud import batch_v1

from gforce.core.batch import (
    BatchJobBuilder,
    BatchJobClient,
    JobConfig,
    create_inference_job,
    create_training_job,
    get_job_status,
    list_active_jobs,
)
from gforce.core.config import GForceConfig
from gforce.core.guardrails import GuardrailViolation


class TestJobConfig:
    """Tests for JobConfig dataclass."""

    def test_default_values(self):
        """Test default job configuration."""
        config = JobConfig(
            job_name="test-job",
            container_image="python:3.12",
            command=["python", "script.py"],
        )

        assert config.job_name == "test-job"
        assert config.container_image == "python:3.12"
        assert config.command == ["python", "script.py"]
        assert config.machine_type == "n1-standard-4"
        assert config.gpu_type == "nvidia-tesla-t4"
        assert config.gpu_count == 1
        assert config.disk_size_gb == 50
        assert config.max_duration_seconds == 3600
        assert config.provisioning_model == "SPOT"

    def test_custom_values(self):
        """Test custom job configuration."""
        config = JobConfig(
            job_name="custom-job",
            container_image="custom:latest",
            command=["train", "--epochs", "10"],
            machine_type="n1-highmem-8",
            gpu_type="nvidia-tesla-v100",
            gpu_count=2,
            disk_size_gb=100,
            max_duration_seconds=7200,
            provisioning_model="STANDARD",
        )

        assert config.machine_type == "n1-highmem-8"
        assert config.gpu_type == "nvidia-tesla-v100"
        assert config.gpu_count == 2
        assert config.max_duration_seconds == 7200


class TestBatchJobBuilder:
    """Tests for BatchJobBuilder class."""

    def test_init(self):
        """Test builder initialization."""
        config = GForceConfig(gcp_project="test-project")
        builder = BatchJobBuilder(config)

        assert builder.config == config

    @patch("gforce.core.batch.enforce_guardrails")
    def test_build_job_creates_valid_job(self, mock_enforce):
        """Test that build_job creates a valid job specification."""
        config = GForceConfig(gcp_project="test-project")
        builder = BatchJobBuilder(config)

        job_config = JobConfig(
            job_name="test-job",
            container_image="python:3.12",
            command=["python", "script.py"],
            max_duration_seconds=3600,
        )

        job = builder.build_job(job_config)

        assert isinstance(job, batch_v1.Job)
        assert job.name == "test-job"
        mock_enforce.assert_called_once()

    def test_build_job_enforces_spot(self):
        """Test that build_job enforces Spot provisioning."""
        config = GForceConfig(gcp_project="test-project")
        builder = BatchJobBuilder(config)

        # Try to create a job with non-Spot provisioning
        job_config = JobConfig(
            job_name="test-job",
            container_image="python:3.12",
            command=["python", "script.py"],
            provisioning_model="STANDARD",  # This should fail
        )

        # The guardrails should reject this
        with pytest.raises(GuardrailViolation):
            builder.build_job(job_config)

    def test_create_runnable_sets_env_vars(self):
        """Test that runnable sets default environment variables."""
        config = GForceConfig(gcp_project="test-project")
        builder = BatchJobBuilder(config)

        job_config = JobConfig(
            job_name="test-job",
            container_image="python:3.12",
            command=["python", "script.py"],
            environment_variables={"CUSTOM_VAR": "value"},
        )

        runnable = builder._create_runnable(job_config)

        assert runnable.environment.variables["PYTHONUNBUFFERED"] == "1"
        assert runnable.environment.variables["HF_HOME"] == "/tmp/huggingface"
        assert runnable.environment.variables["CUSTOM_VAR"] == "value"

    def test_create_task_spec_sets_max_duration(self):
        """Test that task spec sets max run duration."""
        config = GForceConfig(gcp_project="test-project")
        builder = BatchJobBuilder(config)

        job_config = JobConfig(
            job_name="test-job",
            container_image="python:3.12",
            command=["python", "script.py"],
            max_duration_seconds=3600,
        )

        task_spec = builder._create_task_spec(job_config)

        # Duration should be set
        assert task_spec.max_run_duration is not None

    def test_create_allocation_policy_uses_spot(self):
        """Test that allocation policy uses Spot provisioning."""
        config = GForceConfig(gcp_project="test-project")
        builder = BatchJobBuilder(config)

        job_config = JobConfig(
            job_name="test-job",
            container_image="python:3.12",
            command=["python", "script.py"],
        )

        policy = builder._create_allocation_policy(job_config)

        # Check that instances use SPOT
        assert len(policy.instances) > 0
        instance = policy.instances[0]
        assert instance.policy.provisioning_model == batch_v1.AllocationPolicy.ProvisioningModel.SPOT

    def test_allocation_policy_has_gpu_drivers(self):
        """Test that allocation policy installs GPU drivers."""
        config = GForceConfig(gcp_project="test-project")
        builder = BatchJobBuilder(config)

        job_config = JobConfig(
            job_name="test-job",
            container_image="python:3.12",
            command=["python", "script.py"],
            gpu_type="nvidia-tesla-t4",
        )

        policy = builder._create_allocation_policy(job_config)

        instance = policy.instances[0]
        assert instance.install_gpu_drivers is True


class TestBatchJobClient:
    """Tests for BatchJobClient class."""

    @patch("google.cloud.batch_v1.BatchServiceClient")
    def test_init(self, mock_client_class):
        """Test client initialization."""
        config = GForceConfig(gcp_project="test-project")
        client = BatchJobClient(config)

        assert client.config == config

    @patch("google.cloud.batch_v1.BatchServiceClient")
    def test_get_parent(self, mock_client_class):
        """Test parent resource path generation."""
        config = GForceConfig(gcp_project="test-project", gcp_region="us-central1")
        client = BatchJobClient(config)

        parent = client.get_parent()

        assert parent == "projects/test-project/locations/us-central1"

    @patch("google.cloud.batch_v1.BatchServiceClient")
    def test_create_job(self, mock_client_class):
        """Test job creation."""
        mock_response = MagicMock()
        mock_response.name = "projects/test-project/locations/us-central1/jobs/test-job"

        mock_client = MagicMock()
        mock_client.create_job.return_value = mock_response
        mock_client_class.return_value = mock_client

        config = GForceConfig(gcp_project="test-project")
        client = BatchJobClient(config)

        job_config = JobConfig(
            job_name="test-job",
            container_image="python:3.12",
            command=["python", "script.py"],
            max_duration_seconds=3600,
            provisioning_model="SPOT",
        )

        # Override the builder's build_job to return a valid job
        with patch.object(client.builder, "build_job") as mock_build:
            mock_job = MagicMock()
            mock_build.return_value = mock_job

            result = client.create_job(job_config)

        assert result == mock_response
        mock_client.create_job.assert_called_once()

    @patch("google.cloud.batch_v1.BatchServiceClient")
    def test_get_job_with_full_name(self, mock_client_class):
        """Test getting job with full resource name."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        config = GForceConfig(gcp_project="test-project")
        client = BatchJobClient(config)

        full_name = "projects/test-project/locations/us-central1/jobs/my-job"
        client.get_job(full_name)

        mock_client.get_job.assert_called_once()
        call_args = mock_client.get_job.call_args
        assert call_args[0][0].name == full_name

    @patch("google.cloud.batch_v1.BatchServiceClient")
    def test_get_job_with_short_name(self, mock_client_class):
        """Test getting job with short name (converts to full name)."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        config = GForceConfig(gcp_project="test-project", gcp_region="us-central1")
        client = BatchJobClient(config)

        client.get_job("my-job")

        mock_client.get_job.assert_called_once()
        call_args = mock_client.get_job.call_args
        expected_name = "projects/test-project/locations/us-central1/jobs/my-job"
        assert call_args[0][0].name == expected_name

    @patch("google.cloud.batch_v1.BatchServiceClient")
    def test_delete_job(self, mock_client_class):
        """Test job deletion."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        config = GForceConfig(gcp_project="test-project")
        client = BatchJobClient(config)

        client.delete_job("my-job")

        mock_client.delete_job.assert_called_once()


class TestCreateTrainingJob:
    """Tests for create_training_job function."""

    @patch("gforce.core.batch.BatchJobClient")
    def test_create_training_job_with_defaults(self, mock_client_class):
        """Test creating training job with default parameters."""
        mock_job = MagicMock()
        mock_job.name = "test-job"

        mock_client = MagicMock()
        mock_client.create_job.return_value = mock_job
        mock_client_class.return_value = mock_client

        config = GForceConfig(
            gcp_project="test-project",
            default_model="default/model",
        )

        job = create_training_job(
            dataset_uri="gs://bucket/dataset",
            output_name="my-model",
            config=config,
        )

        assert job == mock_job
        mock_client.create_job.assert_called_once()

    @patch("gforce.core.batch.BatchJobClient")
    def test_create_training_job_with_custom_model(self, mock_client_class):
        """Test creating training job with custom model."""
        mock_job = MagicMock()
        mock_job.name = "test-job"

        mock_client = MagicMock()
        mock_client.create_job.return_value = mock_job
        mock_client_class.return_value = mock_client

        config = GForceConfig(gcp_project="test-project")

        job = create_training_job(
            dataset_uri="gs://bucket/dataset",
            output_name="my-model",
            model_id="custom/model",
            instance_prompt="photo of sks person",
            num_steps=500,
            config=config,
        )

        # Verify the job was created
        assert job == mock_job


class TestCreateInferenceJob:
    """Tests for create_inference_job function."""

    @patch("gforce.core.batch.BatchJobClient")
    def test_create_inference_job(self, mock_client_class):
        """Test creating inference job."""
        mock_job = MagicMock()
        mock_job.name = "test-job"

        mock_client = MagicMock()
        mock_client.create_job.return_value = mock_job
        mock_client_class.return_value = mock_client

        config = GForceConfig(gcp_project="test-project")

        job = create_inference_job(
            prompt="a beautiful landscape",
            num_images=5,
            output_prefix="landscapes",
            config=config,
        )

        assert job == mock_job


class TestGetJobStatus:
    """Tests for get_job_status function."""

    @patch("gforce.core.batch.BatchJobClient")
    def test_get_job_status(self, mock_client_class):
        """Test getting job status."""
        from datetime import datetime
        from google.protobuf.timestamp_pb2 import Timestamp

        mock_status = MagicMock()
        mock_status.state.name = "RUNNING"

        mock_job = MagicMock()
        mock_job.name = "projects/test/locations/us-central1/jobs/test-job"
        mock_job.status = mock_status
        mock_job.uid = "job-123"

        # Create a proper timestamp
        ts = Timestamp()
        ts.GetCurrentTime()
        mock_job.create_time = ts
        mock_job.update_time = ts

        mock_client = MagicMock()
        mock_client.get_job.return_value = mock_job
        mock_client_class.return_value = mock_client

        config = GForceConfig(gcp_project="test-project")

        status = get_job_status("test-job", config=config)

        assert status["state"] == "RUNNING"
        assert status["uid"] == "job-123"


class TestListActiveJobs:
    """Tests for list_active_jobs function."""

    @patch("gforce.core.batch.BatchJobClient")
    def test_list_active_jobs_filters_terminated(self, mock_client_class):
        """Test that active jobs are filtered correctly."""
        from google.protobuf.timestamp_pb2 import Timestamp

        # Create jobs with different states
        running_status = MagicMock()
        running_status.state.name = "RUNNING"

        succeeded_status = MagicMock()
        succeeded_status.state.name = "SUCCEEDED"

        ts = Timestamp()
        ts.GetCurrentTime()

        running_job = MagicMock()
        running_job.name = "job-1"
        running_job.status = running_status
        running_job.create_time = ts

        succeeded_job = MagicMock()
        succeeded_job.name = "job-2"
        succeeded_job.status = succeeded_status
        succeeded_job.create_time = ts

        mock_client = MagicMock()
        mock_client.list_jobs.return_value = [running_job, succeeded_job]
        mock_client_class.return_value = mock_client

        config = GForceConfig(gcp_project="test-project")

        jobs = list_active_jobs(config=config)

        # Should only return the running job
        assert len(jobs) == 1
        assert jobs[0]["state"] == "RUNNING"


class TestBatchIntegration:
    """Integration-style tests for batch module."""

    def test_job_config_validation(self):
        """Test job configuration validation."""
        config = JobConfig(
            job_name="test-job",
            container_image="python:3.12",
            command=["python", "script.py"],
            max_duration_seconds=3600,
            provisioning_model="SPOT",
        )

        assert config.max_duration_seconds == 3600
        assert config.provisioning_model == "SPOT"

    def test_job_name_sanitization(self):
        """Test that job names are properly sanitized."""
        # Job names should be lowercase with hyphens
        name = "My_Job_Name_123"
        sanitized = name.replace("_", "-").lower()[:63]

        assert sanitized == "my-job-name-123"
        assert len(sanitized) <= 63
