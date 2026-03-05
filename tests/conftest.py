"""Pytest configuration and fixtures for G-Force tests."""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gforce.core.config import GForceConfig, set_config


@pytest.fixture
def mock_gcp_project():
    """Provide a mock GCP project ID."""
    return "test-project-123456"


@pytest.fixture
def mock_config(mock_gcp_project):
    """Provide a mock GForceConfig."""
    return GForceConfig(
        gcp_project=mock_gcp_project,
        gcp_region="us-central1",
        gcp_zone="us-central1-a",
        bucket_name=f"gforce-assets-{mock_gcp_project}",
    )


@pytest.fixture
def set_mock_config(mock_config):
    """Set the global config to a mock config."""
    original_config = get_config()
    set_config(mock_config)
    yield mock_config
    set_config(original_config)


@pytest.fixture
def mock_adc():
    """Mock Google Application Default Credentials."""
    with patch("gforce.core.auth.google_auth_default") as mock:
        mock_creds = MagicMock()
        mock_creds.expired = False
        mock_creds.valid = True
        mock.return_value = (mock_creds, "test-project-123456")
        yield mock


@pytest.fixture
def mock_storage_client():
    """Mock Google Cloud Storage client."""
    with patch("google.cloud.storage.Client") as mock:
        mock_bucket = MagicMock()
        mock_bucket.exists.return_value = True
        mock_bucket.name = "test-bucket"

        mock_client = MagicMock()
        mock_client.bucket.return_value = mock_bucket

        mock.return_value = mock_client
        yield mock_client


@pytest.fixture
def mock_batch_client():
    """Mock Cloud Batch client."""
    with patch("google.cloud.batch_v1.BatchServiceClient") as mock:
        mock_job = MagicMock()
        mock_job.name = "projects/test/locations/us-central1/jobs/test-job"

        mock_client = MagicMock()
        mock_client.create_job.return_value = mock_job

        mock.return_value = mock_client
        yield mock_client


@pytest.fixture
def temp_dataset(tmp_path):
    """Create a temporary dataset directory with sample images."""
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    # Create dummy image files
    for i in range(5):
        (dataset_dir / f"image_{i:03d}.jpg").write_bytes(b"fake_image_data")

    return dataset_dir


@pytest.fixture
def temp_model_dir(tmp_path):
    """Create a temporary model directory structure."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    # Create dummy model files
    (model_dir / "model_index.json").write_text('{"_class_name": "DummyModel"}')
    (model_dir / "diffusion_pytorch_model.safetensors").write_bytes(b"fake_model_data")

    return model_dir


@pytest.fixture(autouse=True)
def clean_environment():
    """Clean up environment variables after each test."""
    # Store original env vars
    original_env = {
        key: os.environ.get(key)
        for key in os.environ.keys()
        if key.startswith("GFORCE_") or key == "GOOGLE_APPLICATION_CREDENTIALS"
    }

    yield

    # Restore original env vars
    for key, value in original_env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


@pytest.fixture
def mock_pulumi_stack():
    """Mock Pulumi stack operations."""
    with patch("gforce.infra.stack.auto.create_or_select_stack") as mock:
        mock_stack = MagicMock()
        mock_stack.up.return_value = MagicMock(
            outputs={
                "bucket_name": {"value": "test-bucket"},
                "bucket_url": {"value": "gs://test-bucket"},
                "service_account_email": {"value": "sa@test.iam.gserviceaccount.com"},
            }
        )
        mock_stack.preview.return_value = MagicMock()
        mock_stack.destroy.return_value = MagicMock(
            summary=MagicMock(result="succeeded")
        )
        mock_stack.outputs = {
            "bucket_name": {"value": "test-bucket"},
        }

        mock.return_value = mock_stack
        yield mock


@pytest.fixture
def mock_subprocess():
    """Mock subprocess.run."""
    with patch("subprocess.run") as mock:
        mock.return_value = MagicMock(returncode=0, stdout="", stderr="")
        yield mock


@pytest.fixture
def cli_runner():
    """Provide a Click CLI test runner."""
    from typer.testing import CliRunner
    return CliRunner()


# Helper function to get config (avoid circular import issues)
def get_config():
    from gforce.core.config import get_config as _get_config
    return _get_config()
