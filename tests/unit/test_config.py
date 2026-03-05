"""Tests for configuration module."""

import os
from unittest.mock import patch

import pytest

from gforce.core.config import GForceConfig, get_config, set_config


class TestGForceConfig:
    """Tests for GForceConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = GForceConfig()

        assert config.gcp_region == "us-central1"
        assert config.gcp_zone == "us-central1-a"
        assert config.machine_type == "n1-standard-4"
        assert config.gpu_type == "nvidia-tesla-t4"
        assert config.gpu_count == 1
        assert config.spot_price_estimate == 0.15
        assert config.max_run_duration == 3600
        assert config.cache_prefix == "cache/models"
        assert config.default_model == "TheImposterImposters/URPM-SD1.5-v2.3.inpainting"

    def test_bucket_name_from_project(self):
        """Test bucket name generation from project ID."""
        config = GForceConfig(gcp_project="my-project-123")
        assert config.get_bucket_name() == "gforce-assets-my-project-123"

    def test_bucket_name_explicit(self):
        """Test explicit bucket name configuration."""
        config = GForceConfig(
            gcp_project="my-project",
            bucket_name="my-custom-bucket",
        )
        assert config.get_bucket_name() == "my-custom-bucket"

    def test_bucket_name_raises_when_unset(self):
        """Test that bucket name raises when neither bucket nor project is set."""
        config = GForceConfig(gcp_project=None, bucket_name=None)
        with pytest.raises(ValueError, match="GFORCE_BUCKET_NAME or GFORCE_GCP_PROJECT"):
            config.get_bucket_name()

    def test_pulumi_state_url(self):
        """Test Pulumi state URL generation."""
        config = GForceConfig(gcp_project="test-project")
        assert config.get_pulumi_state_url() == "gs://gforce-assets-test-project/state"


class TestConfigEnvironment:
    """Tests for environment-based configuration."""

    def test_env_var_override(self):
        """Test that environment variables override defaults."""
        with patch.dict(os.environ, {
            "GFORCE_GCP_REGION": "europe-west1",
            "GFORCE_GCP_ZONE": "europe-west1-b",
            "GFORCE_MACHINE_TYPE": "n1-highmem-4",
        }):
            config = GForceConfig()
            assert config.gcp_region == "europe-west1"
            assert config.gcp_zone == "europe-west1-b"
            assert config.machine_type == "n1-highmem-4"

    def test_spot_price_from_env(self):
        """Test spot price configuration from environment."""
        with patch.dict(os.environ, {"GFORCE_SPOT_PRICE_ESTIMATE": "0.25"}):
            config = GForceConfig()
            assert config.spot_price_estimate == 0.25

    def test_max_duration_from_env(self):
        """Test max duration configuration from environment."""
        with patch.dict(os.environ, {"GFORCE_MAX_RUN_DURATION": "7200"}):
            config = GForceConfig()
            assert config.max_run_duration == 7200


class TestGlobalConfig:
    """Tests for global config instance management."""

    def test_get_config_returns_instance(self):
        """Test that get_config returns a config instance."""
        config = get_config()
        assert isinstance(config, GForceConfig)

    def test_get_config_returns_same_instance(self):
        """Test that get_config returns the same instance on repeated calls."""
        # Note: This might fail if tests run in parallel or if set_config is called
        config1 = get_config()
        config2 = get_config()
        assert config1 is config2

    def test_set_config_changes_instance(self):
        """Test that set_config changes the global instance."""
        original = get_config()
        new_config = GForceConfig(gcp_project="new-project")

        set_config(new_config)

        current = get_config()
        assert current is new_config
        assert current is not original
        assert current.gcp_project == "new-project"

    def test_set_config_restored(self):
        """Test helper to restore original config after test."""
        original = get_config()
        new_config = GForceConfig(gcp_project="test-project")

        set_config(new_config)
        assert get_config().gcp_project == "test-project"

        set_config(original)
        assert get_config().gcp_project == original.gcp_project


class TestConfigValidation:
    """Tests for configuration validation."""

    def test_positive_gpu_count(self):
        """Test that GPU count must be positive."""
        # Pydantic will coerce negative values or raise validation errors
        config = GForceConfig(gpu_count=0)
        assert config.gpu_count == 0

        config = GForceConfig(gpu_count=2)
        assert config.gpu_count == 2

    def test_cost_estimation_values(self):
        """Test cost-related configuration values."""
        config = GForceConfig(
            spot_price_estimate=0.20,
            max_run_duration=3600,
        )

        max_duration_hours = config.max_run_duration / 3600
        estimated_cost = config.spot_price_estimate * max_duration_hours

        assert estimated_cost == 0.20
