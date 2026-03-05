"""Tests for infrastructure module."""

from unittest.mock import MagicMock, patch

import pytest

from gforce.core.config import GForceConfig
from gforce.infra.stack import (
    PulumiStackManager,
    create_infrastructure,
    deploy_infrastructure,
    destroy_infrastructure,
    get_infrastructure_status,
)


class TestPulumiStackManager:
    """Tests for PulumiStackManager class."""

    def test_init(self):
        """Test manager initialization."""
        config = GForceConfig(
            gcp_project="test-project",
            pulumi_stack="test",
        )
        manager = PulumiStackManager(config)

        assert manager.config == config
        assert manager.project_name == "gforce"
        assert manager.stack_name == "test"

    def test_init_with_default_config(self):
        """Test manager initialization with default config."""
        config = GForceConfig(gcp_project="test-project")
        manager = PulumiStackManager(config)

        assert manager.stack_name == "dev"  # Default stack name

    @patch("gforce.infra.stack.auto.create_or_select_stack")
    def test_preview(self, mock_create_stack):
        """Test preview functionality."""
        mock_stack = MagicMock()
        mock_preview_result = MagicMock()
        mock_stack.preview.return_value = mock_preview_result
        mock_create_stack.return_value = mock_stack

        config = GForceConfig(gcp_project="test-project")
        manager = PulumiStackManager(config)

        result = manager.preview()

        mock_create_stack.assert_called_once()
        mock_stack.set_config.assert_any_call(
            "gcp:region",
            MagicMock(),
        )
        assert result == mock_preview_result

    @patch("gforce.infra.stack.auto.create_or_select_stack")
    def test_up(self, mock_create_stack):
        """Test deployment (up) functionality."""
        mock_stack = MagicMock()
        mock_up_result = MagicMock()
        mock_up_result.outputs = {
            "bucket_name": {"value": "test-bucket"},
            "bucket_url": {"value": "gs://test-bucket"},
        }
        mock_stack.up.return_value = mock_up_result
        mock_create_stack.return_value = mock_stack

        config = GForceConfig(gcp_project="test-project")
        manager = PulumiStackManager(config)

        result = manager.up()

        mock_stack.up.assert_called_once()
        assert result == mock_up_result

    @patch("gforce.infra.stack.auto.create_or_select_stack")
    def test_destroy(self, mock_create_stack):
        """Test destroy functionality."""
        mock_stack = MagicMock()
        mock_destroy_result = MagicMock()
        mock_destroy_result.summary.result = "succeeded"
        mock_stack.destroy.return_value = mock_destroy_result
        mock_create_stack.return_value = mock_stack

        config = GForceConfig(gcp_project="test-project")
        manager = PulumiStackManager(config)

        result = manager.destroy()

        mock_stack.destroy.assert_called_once()
        assert result == mock_destroy_result

    @patch("gforce.infra.stack.auto.create_or_select_stack")
    def test_get_outputs(self, mock_create_stack):
        """Test getting stack outputs."""
        mock_stack = MagicMock()
        mock_outputs = {
            "bucket_name": MagicMock(value="test-bucket"),
        }
        mock_stack.outputs = mock_outputs
        mock_create_stack.return_value = mock_stack

        config = GForceConfig(gcp_project="test-project")
        manager = PulumiStackManager(config)

        outputs = manager.get_outputs()

        assert outputs == mock_outputs


class TestCreateInfrastructure:
    """Tests for create_infrastructure function."""

    @patch("gforce.infra.stack.gcp.storage.Bucket")
    @patch("gforce.infra.stack.gcp.serviceaccount.Account")
    @patch("gforce.infra.stack.gcp.projects.IAMMember")
    @patch("gforce.infra.stack.gcp.compute.Network")
    def test_create_infrastructure_basic(
        self,
        mock_network_class,
        mock_iam_class,
        mock_account_class,
        mock_bucket_class,
    ):
        """Test basic infrastructure creation."""
        mock_bucket = MagicMock()
        mock_bucket_class.return_value = mock_bucket

        mock_account = MagicMock()
        mock_account_class.return_value = mock_account

        mock_network = MagicMock()
        mock_network_class.return_value = mock_network

        config = GForceConfig(
            gcp_project="test-project",
            gcp_region="us-central1",
        )

        resources = create_infrastructure(config)

        assert "bucket" in resources
        assert "service_account" in resources
        assert "network" in resources
        mock_bucket_class.assert_called_once()

    def test_create_infrastructure_raises_without_project(self):
        """Test that infrastructure creation requires project."""
        config = GForceConfig(gcp_project=None)

        with pytest.raises(ValueError, match="GCP project must be configured"):
            create_infrastructure(config)


class TestDeployInfrastructure:
    """Tests for deploy_infrastructure function."""

    @patch("gforce.infra.stack.PulumiStackManager")
    def test_deploy_preview_only(self, mock_manager_class):
        """Test preview-only deployment."""
        mock_manager = MagicMock()
        mock_preview_result = MagicMock()
        mock_manager.preview.return_value = mock_preview_result
        mock_manager_class.return_value = mock_manager

        config = GForceConfig(gcp_project="test-project")

        result = deploy_infrastructure(preview_only=True, config=config)

        mock_manager.preview.assert_called_once()
        mock_manager.up.assert_not_called()
        assert "preview" in result

    @patch("gforce.infra.stack.PulumiStackManager")
    def test_deploy_full(self, mock_manager_class):
        """Test full deployment."""
        mock_up_result = MagicMock()
        mock_up_result.outputs = {
            "bucket_name": {"value": "test-bucket"},
            "bucket_url": {"value": "gs://test-bucket"},
            "service_account_email": {"value": "sa@test.iam.gserviceaccount.com"},
        }

        mock_manager = MagicMock()
        mock_manager.up.return_value = mock_up_result
        mock_manager_class.return_value = mock_manager

        config = GForceConfig(gcp_project="test-project")

        result = deploy_infrastructure(preview_only=False, config=config)

        mock_manager.up.assert_called_once()
        assert result["bucket_name"] == "test-bucket"
        assert result["bucket_url"] == "gs://test-bucket"


class TestDestroyInfrastructure:
    """Tests for destroy_infrastructure function."""

    @patch("gforce.infra.stack.PulumiStackManager")
    def test_destroy_success(self, mock_manager_class):
        """Test successful destruction."""
        mock_destroy_result = MagicMock()
        mock_destroy_result.summary.result = "succeeded"

        mock_manager = MagicMock()
        mock_manager.destroy.return_value = mock_destroy_result
        mock_manager_class.return_value = mock_manager

        config = GForceConfig(gcp_project="test-project")

        result = destroy_infrastructure(force=True, config=config)

        mock_manager.destroy.assert_called_once()
        assert result["destroyed"] is True

    @patch("gforce.infra.stack.PulumiStackManager")
    def test_destroy_failure(self, mock_manager_class):
        """Test failed destruction."""
        mock_destroy_result = MagicMock()
        mock_destroy_result.summary.result = "failed"

        mock_manager = MagicMock()
        mock_manager.destroy.return_value = mock_destroy_result
        mock_manager_class.return_value = mock_manager

        config = GForceConfig(gcp_project="test-project")

        result = destroy_infrastructure(force=True, config=config)

        assert result["destroyed"] is False


class TestGetInfrastructureStatus:
    """Tests for get_infrastructure_status function."""

    @patch("gforce.infra.stack.PulumiStackManager")
    def test_get_status(self, mock_manager_class):
        """Test getting infrastructure status."""
        mock_outputs = {
            "bucket_name": MagicMock(value="test-bucket"),
            "bucket_url": MagicMock(value="gs://test-bucket"),
            "service_account_email": MagicMock(value="sa@test.com"),
        }

        mock_manager = MagicMock()
        mock_manager.get_outputs.return_value = mock_outputs
        mock_manager_class.return_value = mock_manager

        config = GForceConfig(gcp_project="test-project")

        status = get_infrastructure_status(config)

        assert status["bucket_name"] == "test-bucket"
        assert status["bucket_url"] == "gs://test-bucket"
        assert status["service_account_email"] == "sa@test.com"


class TestInfraIntegration:
    """Integration-style tests for infra module."""

    def test_stack_manager_workflow(self):
        """Test complete stack manager workflow."""
        config = GForceConfig(
            gcp_project="test-project",
            gcp_region="us-central1",
            pulumi_stack="dev",
        )

        manager = PulumiStackManager(config)

        assert manager.project_name == "gforce"
        assert manager.stack_name == "dev"
        assert manager.work_dir == "."

    def test_bucket_name_in_resources(self):
        """Test that bucket name is correctly used in resource creation."""
        config = GForceConfig(
            gcp_project="test-project",
            gcp_region="us-central1",
        )

        bucket_name = config.get_bucket_name()
        assert bucket_name == "gforce-assets-test-project"

    @patch("gforce.infra.stack.gcp.storage.Bucket")
    def test_lifecycle_rules_configuration(self, mock_bucket_class):
        """Test that lifecycle rules are properly configured."""
        mock_bucket = MagicMock()
        mock_bucket_class.return_value = mock_bucket

        config = GForceConfig(gcp_project="test-project")

        with patch("gforce.infra.stack.gcp.serviceaccount.Account"):
            with patch("gforce.infra.stack.gcp.projects.IAMMember"):
                with patch("gforce.infra.stack.gcp.compute.Network"):
                    create_infrastructure(config)

        # Verify bucket was created with lifecycle rules
        mock_bucket_class.assert_called_once()
        call_kwargs = mock_bucket_class.call_args[1]
        assert "lifecycle_rules" in call_kwargs
