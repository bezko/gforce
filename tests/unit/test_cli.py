"""Tests for CLI module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from gforce.cli.main import app

runner = CliRunner()


class TestMainCallback:
    """Tests for main CLI callback."""

    def test_version_flag(self):
        """Test that version can be checked."""
        # Note: Typer doesn't have built-in --version, so we just test help
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "G-Force CLI" in result.output


class TestAuthStatus:
    """Tests for auth-status command."""

    @patch("gforce.cli.main.validate_adc")
    def test_auth_status_authorized(self, mock_validate):
        """Test auth status when authorized."""
        mock_validate.return_value = (MagicMock(), "test-project")

        result = runner.invoke(app, ["auth-status"])

        assert result.exit_code == 0
        assert "Authenticated" in result.output or "project" in result.output

    @patch("gforce.cli.main.validate_adc")
    def test_auth_status_unauthorized(self, mock_validate):
        """Test auth status when not authorized."""
        from gforce.core.auth import AuthenticationError

        mock_validate.side_effect = AuthenticationError("Invalid credentials")

        result = runner.invoke(app, ["auth-status"])

        # Should still exit 0 but show not authenticated
        assert result.exit_code == 0


class TestInitCommand:
    """Tests for init command."""

    @patch("gforce.cli.main.get_project_id")
    @patch("gforce.cli.main.bootstrap_project")
    @patch("gforce.cli.main.deploy_infrastructure")
    def test_init_success(
        self,
        mock_deploy,
        mock_bootstrap,
        mock_get_project,
    ):
        """Test successful initialization."""
        mock_get_project.return_value = "test-project"
        mock_bootstrap.return_value = {
            "project_id": "test-project",
            "bucket_name": "gforce-assets-test-project",
            "apis_enabled": ["batch.googleapis.com"],
            "bucket_created": True,
            "pulumi_configured": True,
        }
        mock_deploy.return_value = {
            "bucket_name": "gforce-assets-test-project",
            "bucket_url": "gs://gforce-assets-test-project",
        }

        result = runner.invoke(app, ["init", "--yes"])

        assert result.exit_code == 0
        mock_bootstrap.assert_called_once()

    @patch("gforce.cli.main.get_project_id")
    def test_init_auth_failure(self, mock_get_project):
        """Test init when authentication fails."""
        from gforce.core.auth import AuthenticationError

        mock_get_project.side_effect = AuthenticationError("Invalid credentials")

        result = runner.invoke(app, ["init"])

        assert result.exit_code == 1
        assert "gcloud auth" in result.output


class TestInfraUpCommand:
    """Tests for infra-up command."""

    @patch("gforce.cli.main.require_auth")
    @patch("gforce.cli.main.deploy_infrastructure")
    def test_infra_up_success(self, mock_deploy, mock_require_auth):
        """Test successful infrastructure deployment."""
        mock_deploy.return_value = {
            "bucket_url": "gs://test-bucket",
        }

        result = runner.invoke(app, ["infra-up"])

        assert result.exit_code == 0
        mock_deploy.assert_called_once()

    @patch("gforce.cli.main.require_auth")
    @patch("gforce.cli.main.deploy_infrastructure")
    def test_infra_up_preview(self, mock_deploy, mock_require_auth):
        """Test infrastructure preview."""
        mock_deploy.return_value = {"preview": "..."}

        result = runner.invoke(app, ["infra-up", "--preview"])

        assert result.exit_code == 0
        mock_deploy.assert_called_once_with(preview_only=True, config=MagicMock())


class TestInfraDownCommand:
    """Tests for infra-down command."""

    @patch("gforce.cli.main.require_auth")
    @patch("gforce.cli.main.destroy_infrastructure")
    def test_infra_down_force(self, mock_destroy, mock_require_auth):
        """Test infrastructure destruction with force flag."""
        mock_destroy.return_value = {"destroyed": True}

        result = runner.invoke(app, ["infra-down", "--force"])

        assert result.exit_code == 0
        mock_destroy.assert_called_once()

    @patch("gforce.cli.main.require_auth")
    @patch("gforce.cli.main.destroy_infrastructure")
    def test_infra_down_confirmation(self, mock_destroy, mock_require_auth):
        """Test infrastructure destruction with confirmation."""
        mock_destroy.return_value = {"destroyed": True}

        result = runner.invoke(app, ["infra-down"], input="destroy\n")

        assert result.exit_code == 0

    @patch("gforce.cli.main.require_auth")
    def test_infra_down_cancel(self, mock_require_auth):
        """Test cancelling infrastructure destruction."""
        result = runner.invoke(app, ["infra-down"], input="no\n")

        assert "Cancelled" in result.output


class TestTrainCommand:
    """Tests for train command."""

    @patch("gforce.cli.main.require_auth")
    @patch("gforce.cli.main.get_spot_price_estimate")
    @patch("gforce.cli.main.display_cost_confirmation")
    @patch("gforce.cli.main.create_training_job")
    def test_train_success(
        self,
        mock_create,
        mock_confirm,
        mock_estimate,
        mock_require_auth,
    ):
        """Test successful training job creation."""
        from gforce.core.guardrails import CostEstimate

        mock_estimate.return_value = CostEstimate(
            hourly_rate=0.15,
            max_duration_hours=1.0,
            estimated_max_cost=0.15,
            is_spot=True,
        )
        mock_confirm.return_value = True
        mock_create.return_value = MagicMock(name="projects/test/locations/us-central1/jobs/job-123")

        result = runner.invoke(app, [
            "train",
            "gs://bucket/dataset",
            "--output", "my-model",
            "--yes",
        ])

        assert result.exit_code == 0
        mock_create.assert_called_once()

    @patch("gforce.cli.main.require_auth")
    def test_train_invalid_dataset(self, mock_require_auth):
        """Test training with invalid dataset URI."""
        result = runner.invoke(app, [
            "train",
            "/local/path",  # Not a gs:// URI
            "--output", "my-model",
        ])

        assert result.exit_code == 1
        assert "gs://" in result.output

    @patch("gforce.cli.main.require_auth")
    @patch("gforce.cli.main.get_spot_price_estimate")
    @patch("gforce.cli.main.display_cost_confirmation")
    def test_train_cancelled(
        self,
        mock_confirm,
        mock_estimate,
        mock_require_auth,
    ):
        """Test training job creation when user cancels."""
        from gforce.core.guardrails import CostEstimate

        mock_estimate.return_value = CostEstimate(
            hourly_rate=0.15,
            max_duration_hours=1.0,
            estimated_max_cost=0.15,
            is_spot=True,
        )
        mock_confirm.return_value = False

        result = runner.invoke(app, [
            "train",
            "gs://bucket/dataset",
            "--output", "my-model",
        ])

        assert result.exit_code == 0
        assert "Cancelled" in result.output


class TestGenCommand:
    """Tests for gen command."""

    @patch("gforce.cli.main.require_auth")
    @patch("gforce.cli.main.get_spot_price_estimate")
    @patch("gforce.cli.main.display_cost_confirmation")
    @patch("gforce.cli.main.create_inference_job")
    def test_gen_success(
        self,
        mock_create,
        mock_confirm,
        mock_estimate,
        mock_require_auth,
    ):
        """Test successful inference job creation."""
        from gforce.core.guardrails import CostEstimate

        mock_estimate.return_value = CostEstimate(
            hourly_rate=0.15,
            max_duration_hours=1.0,
            estimated_max_cost=0.15,
            is_spot=True,
        )
        mock_confirm.return_value = True
        mock_create.return_value = MagicMock(name="projects/test/locations/us-central1/jobs/job-123")

        result = runner.invoke(app, [
            "gen",
            "a beautiful landscape",
            "--num-images", "5",
            "--yes",
        ])

        assert result.exit_code == 0
        mock_create.assert_called_once()


class TestStatusCommand:
    """Tests for status command."""

    @patch("gforce.cli.main.require_auth")
    @patch("gforce.cli.main.list_active_jobs")
    def test_status_list(self, mock_list, mock_require_auth):
        """Test listing active jobs."""
        mock_list.return_value = [
            {
                "name": "job-1",
                "state": "RUNNING",
                "create_time": "2024-01-01T00:00:00",
            }
        ]

        result = runner.invoke(app, ["status"])

        assert result.exit_code == 0
        assert "RUNNING" in result.output

    @patch("gforce.cli.main.require_auth")
    @patch("gforce.cli.main.get_job_status")
    def test_status_specific_job(self, mock_get_status, mock_require_auth):
        """Test getting status of specific job."""
        mock_get_status.return_value = {
            "name": "projects/test/locations/us-central1/jobs/my-job",
            "state": "SUCCEEDED",
            "uid": "job-123",
        }

        result = runner.invoke(app, ["status", "my-job"])

        assert result.exit_code == 0
        assert "SUCCEEDED" in result.output


class TestPullCommand:
    """Tests for pull command."""

    @patch("gforce.cli.main.require_auth")
    @patch("subprocess.run")
    def test_pull_success(self, mock_run, mock_require_auth):
        """Test successful download."""
        mock_run.return_value = MagicMock(returncode=0)

        with runner.isolated_filesystem() as tmp:
            result = runner.invoke(app, ["pull", "my-output", tmp])

            assert result.exit_code == 0
            mock_run.assert_called_once()

    @patch("gforce.cli.main.require_auth")
    @patch("subprocess.run")
    def test_pull_failure(self, mock_run, mock_require_auth):
        """Test handling of download failure."""
        mock_run.return_value = MagicMock(returncode=1, stderr="Access denied")

        with runner.isolated_filesystem() as tmp:
            result = runner.invoke(app, ["pull", "my-output", tmp])

            assert result.exit_code == 1


class TestCacheListCommand:
    """Tests for cache-list command."""

    @patch("gforce.cli.main.require_auth")
    @patch("gforce.cli.main.ModelCache")
    def test_cache_list(self, mock_cache_class, mock_require_auth):
        """Test listing cached models."""
        mock_blob1 = MagicMock()
        mock_blob1.name = "cache/models/huggingface/user--model1/manifest.json"
        mock_blob2 = MagicMock()
        mock_blob2.name = "cache/models/huggingface/user--model2/manifest.json"

        mock_cache = MagicMock()
        mock_cache.client.list_blobs.return_value = [mock_blob1, mock_blob2]
        mock_cache_class.return_value = mock_cache

        result = runner.invoke(app, ["cache-list"])

        assert result.exit_code == 0


class TestCLIIntegration:
    """Integration-style tests for CLI."""

    def test_help_output(self):
        """Test that help output is formatted correctly."""
        result = runner.invoke(app, ["--help"])

        assert result.exit_code == 0
        assert "G-Force" in result.output

        # Check that all commands are listed
        commands = ["init", "infra-up", "infra-down", "train", "gen", "status", "pull"]
        for cmd in commands:
            assert cmd in result.output

    def test_command_help_individually(self):
        """Test individual command help."""
        commands = [
            ["init", "--help"],
            ["train", "--help"],
            ["gen", "--help"],
            ["status", "--help"],
            ["pull", "--help"],
        ]

        for cmd in commands:
            result = runner.invoke(app, cmd)
            assert result.exit_code == 0, f"Help failed for {cmd[0]}"
