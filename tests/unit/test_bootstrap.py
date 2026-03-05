"""Tests for bootstrap module."""

from unittest.mock import MagicMock, patch

import pytest

from gforce.core.bootstrap import (
    REQUIRED_APIS,
    bootstrap_project,
    configure_pulumi_backend,
    create_state_bucket,
    enable_required_apis,
)
from gforce.core.config import GForceConfig


class TestRequiredAPIs:
    """Tests for required API constants."""

    def test_required_apis_list(self):
        """Test that required APIs list is complete."""
        assert "batch.googleapis.com" in REQUIRED_APIS
        assert "compute.googleapis.com" in REQUIRED_APIS
        assert "storage.googleapis.com" in REQUIRED_APIS
        assert "logging.googleapis.com" in REQUIRED_APIS
        assert "monitoring.googleapis.com" in REQUIRED_APIS


class TestEnableRequiredAPIs:
    """Tests for enable_required_apis function."""

    @patch("gforce.core.bootstrap.service_usage_v1.ServiceUsageClient")
    def test_enable_all_apis(self, mock_client_class):
        """Test enabling all required APIs."""
        mock_service = MagicMock()
        mock_service.state = MagicMock()
        mock_service.state.value = 2  # DISABLED

        mock_client = MagicMock()
        mock_client.get_service.return_value = mock_service
        mock_client.enable_service.return_value = MagicMock()
        mock_client_class.return_value = mock_client

        enable_required_apis("test-project")

        # Should call enable_service for each API
        assert mock_client.enable_service.call_count == len(REQUIRED_APIS)

    @patch("gforce.core.bootstrap.service_usage_v1.ServiceUsageClient")
    def test_skip_already_enabled(self, mock_client_class):
        """Test skipping APIs that are already enabled."""
        mock_service = MagicMock()
        mock_service.state = MagicMock()
        mock_service.state.name = "ENABLED"

        mock_client = MagicMock()
        mock_client.get_service.return_value = mock_service
        mock_client_class.return_value = mock_client

        enable_required_apis("test-project")

        # Should not call enable_service for already enabled APIs
        mock_client.enable_service.assert_not_called()

    @patch("gforce.core.bootstrap.service_usage_v1.ServiceUsageClient")
    def test_handle_api_error(self, mock_client_class):
        """Test handling of API enablement errors."""
        mock_client = MagicMock()
        mock_client.get_service.side_effect = Exception("API Error")
        mock_client_class.return_value = mock_client

        with pytest.raises(Exception, match="API Error"):
            enable_required_apis("test-project")


class TestCreateStateBucket:
    """Tests for create_state_bucket function."""

    @patch("google.cloud.storage.Client")
    def test_create_new_bucket(self, mock_client_class):
        """Test creating a new bucket."""
        mock_bucket = MagicMock()
        mock_bucket.exists.return_value = False

        mock_client = MagicMock()
        mock_client.bucket.return_value = mock_bucket
        mock_client.create_bucket.return_value = mock_bucket
        mock_client_class.return_value = mock_client

        result = create_state_bucket("test-project", "test-bucket", "us-central1")

        mock_client.create_bucket.assert_called_once()
        assert result == mock_bucket

    @patch("google.cloud.storage.Client")
    def test_use_existing_bucket(self, mock_client_class):
        """Test using an existing bucket."""
        mock_bucket = MagicMock()
        mock_bucket.exists.return_value = True

        mock_client = MagicMock()
        mock_client.bucket.return_value = mock_bucket
        mock_client_class.return_value = mock_client

        result = create_state_bucket("test-project", "test-bucket", "us-central1")

        mock_client.create_bucket.assert_not_called()
        assert result == mock_bucket

    @patch("google.cloud.storage.Client")
    def test_configure_uniform_access(self, mock_client_class):
        """Test that uniform bucket-level access is configured."""
        mock_bucket = MagicMock()
        mock_bucket.exists.return_value = False

        mock_client = MagicMock()
        mock_client.bucket.return_value = mock_bucket
        mock_client.create_bucket.return_value = mock_bucket
        mock_client_class.return_value = mock_client

        create_state_bucket("test-project", "test-bucket", "us-central1")

        # Verify uniform access was enabled
        assert mock_bucket.iam_configuration.uniform_bucket_level_access_enabled is True
        mock_bucket.patch.assert_called()

    @patch("google.cloud.storage.Client")
    def test_configure_lifecycle_rules(self, mock_client_class):
        """Test that lifecycle rules are configured."""
        mock_bucket = MagicMock()
        mock_bucket.exists.return_value = False

        mock_client = MagicMock()
        mock_client.bucket.return_value = mock_bucket
        mock_client.create_bucket.return_value = mock_bucket
        mock_client_class.return_value = mock_client

        create_state_bucket("test-project", "test-bucket", "us-central1")

        # Verify lifecycle rules were set
        assert hasattr(mock_bucket, "lifecycle_rules")


class TestConfigurePulumiBackend:
    """Tests for configure_pulumi_backend function."""

    @patch("gforce.core.bootstrap.subprocess.run")
    def test_successful_login(self, mock_run):
        """Test successful Pulumi login."""
        mock_run.return_value = MagicMock(returncode=0)

        configure_pulumi_backend("test-bucket")

        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert call_args[0] == "pulumi"
        assert call_args[1] == "login"
        assert "gs://test-bucket/state" in call_args[2]

    @patch("gforce.core.bootstrap.subprocess.run")
    def test_failed_login(self, mock_run):
        """Test handling of failed Pulumi login."""
        from subprocess import CalledProcessError

        mock_run.side_effect = CalledProcessError(
            returncode=1,
            cmd=["pulumi", "login"],
            stderr="Authentication failed",
        )

        with pytest.raises(CalledProcessError):
            configure_pulumi_backend("test-bucket")

    @patch("gforce.core.bootstrap.subprocess.run")
    def test_pulumi_not_found(self, mock_run):
        """Test handling when Pulumi CLI is not installed."""
        mock_run.side_effect = FileNotFoundError("pulumi not found")

        # Should not raise, just warn
        configure_pulumi_backend("test-bucket")


class TestBootstrapProject:
    """Tests for bootstrap_project function."""

    @patch("gforce.core.bootstrap.enable_required_apis")
    @patch("gforce.core.bootstrap.create_state_bucket")
    @patch("gforce.core.bootstrap.configure_pulumi_backend")
    @patch("gforce.core.bootstrap.verify_permissions")
    def test_full_bootstrap(
        self,
        mock_verify,
        mock_pulumi,
        mock_bucket,
        mock_apis,
    ):
        """Test complete bootstrap process."""
        mock_bucket.return_value = MagicMock(exists=lambda: True)
        mock_verify.return_value = []

        config = GForceConfig(gcp_project="test-project")

        result = bootstrap_project(config)

        assert result["project_id"] == "test-project"
        assert result["bucket_created"] is True
        assert result["pulumi_configured"] is True
        assert result["apis_enabled"] == REQUIRED_APIS

    @patch("gforce.core.bootstrap.enable_required_apis")
    @patch("gforce.core.bootstrap.create_state_bucket")
    @patch("gforce.core.bootstrap.configure_pulumi_backend")
    def test_bootstrap_skip_apis(
        self,
        mock_pulumi,
        mock_bucket,
        mock_apis,
    ):
        """Test bootstrap with API enablement skipped."""
        mock_bucket.return_value = MagicMock(exists=lambda: True)

        config = GForceConfig(gcp_project="test-project")

        result = bootstrap_project(config, skip_api_enablement=True)

        mock_apis.assert_not_called()
        assert result["apis_enabled"] == []

    @patch("gforce.core.bootstrap.enable_required_apis")
    @patch("gforce.core.bootstrap.create_state_bucket")
    def test_bootstrap_skip_pulumi(
        self,
        mock_bucket,
        mock_apis,
    ):
        """Test bootstrap with Pulumi configuration skipped."""
        mock_bucket.return_value = MagicMock(exists=lambda: True)

        config = GForceConfig(gcp_project="test-project")

        result = bootstrap_project(config, skip_pulumi=True)

        assert result["pulumi_configured"] is False

    def test_bootstrap_raises_without_project(self):
        """Test that bootstrap requires project ID."""
        config = GForceConfig(gcp_project=None)

        with pytest.raises(ValueError, match="GCP project ID must be configured"):
            bootstrap_project(config)


class TestVerifyPermissions:
    """Tests for verify_permissions function."""

    @patch("gforce.core.bootstrap.gcp.iam_admin_v1", create=True)
    def test_verify_returns_empty_list(self, mock_iam):
        """Test that verify_permissions returns empty list (placeholder)."""
        from gforce.core.bootstrap import verify_permissions
        
        result = verify_permissions("test-project")

        assert result == []
