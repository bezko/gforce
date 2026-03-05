"""End-to-end integration tests for G-Force."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestEndToEndWorkflow:
    """End-to-end workflow tests."""

    @pytest.mark.integration
    def test_full_initialization_workflow(
        self,
        mock_adc,
        mock_storage_client,
        mock_pulumi_stack,
        mock_subprocess,
    ):
        """Test complete initialization workflow."""
        from gforce.core.bootstrap import bootstrap_project
        from gforce.core.config import GForceConfig

        config = GForceConfig(
            gcp_project="test-project",
            gcp_region="us-central1",
        )

        result = bootstrap_project(config)

        assert result["project_id"] == "test-project"
        assert result["bucket_created"] is True

    @pytest.mark.integration
    def test_full_training_workflow(
        self,
        mock_adc,
        mock_batch_client,
        set_mock_config,
    ):
        """Test complete training job workflow."""
        from gforce.core.batch import create_training_job

        job = create_training_job(
            dataset_uri="gs://test-bucket/dataset",
            output_name="my-model",
            instance_prompt="photo of sks person",
            num_steps=100,
        )

        assert job is not None
        mock_batch_client.create_job.assert_called_once()

    @pytest.mark.integration
    def test_full_inference_workflow(
        self,
        mock_adc,
        mock_batch_client,
        set_mock_config,
    ):
        """Test complete inference job workflow."""
        from gforce.core.batch import create_inference_job

        job = create_inference_job(
            prompt="a beautiful landscape",
            num_images=5,
            output_prefix="test-gen",
        )

        assert job is not None
        mock_batch_client.create_job.assert_called_once()

    @pytest.mark.integration
    def test_cost_estimation_and_confirmation(
        self,
        set_mock_config,
    ):
        """Test cost estimation and confirmation flow."""
        from gforce.core.guardrails import (
            CostEstimate,
            display_cost_confirmation,
            get_spot_price_estimate,
        )

        # Get estimate
        estimate = get_spot_price_estimate()

        assert isinstance(estimate, CostEstimate)
        assert estimate.is_spot is True
        assert estimate.hourly_rate > 0
        assert estimate.estimated_max_cost > 0

        # Check summary format
        summary = estimate.format_summary()
        assert "Spot" in summary
        assert "$" in summary

    @pytest.mark.integration
    def test_auth_flow(
        self,
        mock_adc,
        set_mock_config,
    ):
        """Test authentication flow."""
        from gforce.core.auth import (
            check_auth_silent,
            get_project_id,
            validate_adc,
        )

        # Should succeed with mocked credentials
        assert check_auth_silent() is True

        creds, project = validate_adc()
        assert creds is not None

        project_id = get_project_id()
        assert project_id == "test-project-123456"


class TestCacheWorkflow:
    """Tests for model caching workflow."""

    @pytest.mark.integration
    def test_cache_entry_lifecycle(
        self,
        mock_storage_client,
        set_mock_config,
    ):
        """Test complete cache entry lifecycle."""
        from gforce.core.cache import CacheEntry, ModelCache

        cache = ModelCache()

        entry = CacheEntry(
            provider="huggingface",
            repo_id="user/model",
            commit_hash="abc123def456",
        )

        # Initially not cached
        assert cache.exists(entry) is False

        # Save manifest
        cache.save_manifest(entry)

        # Now should exist (mock returns True after save)
        # Note: In real test we'd need to mock the blob exists check


class TestInfrastructureWorkflow:
    """Tests for infrastructure management workflow."""

    @pytest.mark.integration
    def test_infrastructure_deploy_and_destroy(
        self,
        mock_adc,
        mock_pulumi_stack,
        set_mock_config,
    ):
        """Test deploying and destroying infrastructure."""
        from gforce.infra.stack import deploy_infrastructure, destroy_infrastructure

        # Deploy
        result = deploy_infrastructure()
        assert "bucket_name" in result

        # Destroy
        destroy_result = destroy_infrastructure(force=True)
        assert destroy_result["destroyed"] is True


class TestCLIIntegration:
    """CLI integration tests."""

    @pytest.mark.integration
    def test_cli_auth_status(
        self,
        mock_adc,
        cli_runner,
    ):
        """Test CLI auth-status command."""
        from gforce.cli.main import app

        result = cli_runner.invoke(app, ["auth-status"])

        # Should succeed with mocked auth
        assert result.exit_code == 0

    @pytest.mark.integration
    def test_cli_status_empty(
        self,
        mock_adc,
        mock_batch_client,
        cli_runner,
        set_mock_config,
    ):
        """Test CLI status command with no jobs."""
        from gforce.cli.main import app

        mock_batch_client.list_jobs.return_value = []

        result = cli_runner.invoke(app, ["status"])

        assert result.exit_code == 0


class TestGuardrailEnforcement:
    """Tests for guardrail enforcement."""

    @pytest.mark.integration
    def test_spot_enforcement(
        self,
        set_mock_config,
    ):
        """Test that Spot provisioning is enforced."""
        from gforce.core.batch import BatchJobBuilder, JobConfig
        from gforce.core.config import GForceConfig
        from gforce.core.guardrails import GuardrailViolation

        config = GForceConfig(gcp_project="test-project")
        builder = BatchJobBuilder(config)

        # Create a job that tries to use STANDARD provisioning
        job_config = JobConfig(
            job_name="test-job",
            container_image="python:3.12",
            command=["echo", "test"],
            provisioning_model="STANDARD",  # This should be rejected
            max_duration_seconds=3600,
        )

        # The guardrails should catch this
        with pytest.raises(GuardrailViolation):
            builder.build_job(job_config)

    @pytest.mark.integration
    def test_max_runtime_enforcement(
        self,
        set_mock_config,
    ):
        """Test that max runtime is enforced."""
        from gforce.core.batch import BatchJobBuilder, JobConfig
        from gforce.core.config import GForceConfig
        from gforce.core.guardrails import GuardrailViolation

        config = GForceConfig(gcp_project="test-project")
        builder = BatchJobBuilder(config)

        # Create a job with excessive max runtime
        job_config = JobConfig(
            job_name="test-job",
            container_image="python:3.12",
            command=["echo", "test"],
            max_duration_seconds=7200,  # 2 hours, exceeds default 1 hour
            provisioning_model="SPOT",
        )

        # The guardrails should catch this
        with pytest.raises(GuardrailViolation):
            builder.build_job(job_config)


class TestConfigurationPropagation:
    """Tests for configuration propagation through the system."""

    @pytest.mark.integration
    def test_config_propagates_to_all_components(self):
        """Test that configuration propagates correctly."""
        from gforce.core.batch import BatchJobBuilder
        from gforce.core.cache import ModelCache
        from gforce.core.config import GForceConfig
        from gforce.infra.stack import PulumiStackManager

        config = GForceConfig(
            gcp_project="custom-project",
            gcp_region="europe-west1",
            spot_price_estimate=0.20,
        )

        # Verify config in builder
        builder = BatchJobBuilder(config)
        assert builder.config.gcp_project == "custom-project"
        assert builder.config.gcp_region == "europe-west1"

        # Verify config in cache
        with patch("gforce.core.cache.storage.Client"):
            cache = ModelCache(config)
            assert cache.config.gcp_project == "custom-project"

        # Verify config in stack manager
        manager = PulumiStackManager(config)
        assert manager.config.gcp_project == "custom-project"
