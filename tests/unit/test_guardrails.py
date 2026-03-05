"""Tests for guardrails module."""

from unittest.mock import MagicMock, patch

import pytest

from gforce.core.config import GForceConfig
from gforce.core.guardrails import (
    CostEstimate,
    GuardrailViolation,
    check_bucket_lifecycle,
    enforce_guardrails,
    get_spot_price_estimate,
    validate_max_runtime,
    validate_spot_configuration,
)


class TestCostEstimate:
    """Tests for CostEstimate dataclass."""

    def test_format_summary(self):
        """Test cost summary formatting."""
        estimate = CostEstimate(
            hourly_rate=0.15,
            max_duration_hours=1.0,
            estimated_max_cost=0.15,
            is_spot=True,
        )

        summary = estimate.format_summary()

        assert "Spot Preemptible" in summary
        assert "$0.15/hour" in summary
        assert "1.0 hours" in summary
        assert "$0.15 USD" in summary

    def test_on_demand_formatting(self):
        """Test formatting for on-demand estimate."""
        estimate = CostEstimate(
            hourly_rate=0.50,
            max_duration_hours=2.0,
            estimated_max_cost=1.00,
            is_spot=False,
        )

        summary = estimate.format_summary()

        assert "On-Demand" in summary
        assert "$0.50/hour" in summary
        assert "$1.00 USD" in summary


class TestGetSpotPriceEstimate:
    """Tests for get_spot_price_estimate function."""

    def test_default_estimate(self):
        """Test default cost estimate."""
        config = GForceConfig(spot_price_estimate=0.15, max_run_duration=3600)

        estimate = get_spot_price_estimate(config=config)

        assert estimate.hourly_rate == 0.15
        assert estimate.max_duration_hours == 1.0
        assert estimate.estimated_max_cost == 0.15
        assert estimate.is_spot is True

    def test_custom_config(self):
        """Test estimate with custom configuration."""
        config = GForceConfig(spot_price_estimate=0.20, max_run_duration=7200)

        estimate = get_spot_price_estimate(config=config)

        assert estimate.hourly_rate == 0.20
        assert estimate.max_duration_hours == 2.0
        assert estimate.estimated_max_cost == 0.40

    def test_region_parameter(self):
        """Test that region parameter is accepted."""
        config = GForceConfig()

        estimate = get_spot_price_estimate(region="europe-west1", config=config)

        assert estimate is not None


class TestValidateSpotConfiguration:
    """Tests for validate_spot_configuration function."""

    def test_valid_spot_configuration(self):
        """Test validation passes with proper Spot config."""
        job_spec = {
            "allocationPolicy": {
                "instances": [{
                    "policy": {
                        "provisioningModel": "SPOT",
                    }
                }]
            },
            "taskGroups": [{}],
        }

        # Should not raise
        validate_spot_configuration(job_spec)

    def test_missing_spot_configuration(self):
        """Test validation fails without Spot config."""
        job_spec = {
            "allocationPolicy": {
                "instances": [{}]
            },
            "taskGroups": [{}],
        }

        with pytest.raises(GuardrailViolation) as exc_info:
            validate_spot_configuration(job_spec)

        assert "SPOT" in str(exc_info.value)

    def test_wrong_provisioning_model(self):
        """Test validation fails with non-Spot provisioning."""
        job_spec = {
            "allocationPolicy": {
                "instances": [{
                    "policy": {
                        "provisioningModel": "STANDARD",
                    }
                }]
            },
            "taskGroups": [{}],
        }

        with pytest.raises(GuardrailViolation) as exc_info:
            validate_spot_configuration(job_spec)

        assert "SPOT" in str(exc_info.value)


class TestValidateMaxRuntime:
    """Tests for validate_max_runtime function."""

    def test_valid_max_runtime(self):
        """Test validation passes with valid max runtime."""
        job_spec = {
            "taskGroups": [{
                "taskSpec": {
                    "maxRunDuration": "3600s",
                }
            }],
        }

        # Should not raise
        validate_max_runtime(job_spec)

    def test_missing_max_runtime(self):
        """Test validation fails without max runtime."""
        job_spec = {
            "taskGroups": [{
                "taskSpec": {}
            }],
        }

        with pytest.raises(GuardrailViolation) as exc_info:
            validate_max_runtime(job_spec)

        assert "maxRunDuration" in str(exc_info.value)

    def test_excessive_max_runtime(self):
        """Test validation fails with excessive max runtime."""
        job_spec = {
            "taskGroups": [{
                "taskSpec": {
                    "maxRunDuration": "7200s",
                }
            }],
        }

        with pytest.raises(GuardrailViolation) as exc_info:
            validate_max_runtime(job_spec, max_seconds=3600)

        assert "exceeds limit" in str(exc_info.value)

    def test_custom_max_seconds(self):
        """Test validation with custom max seconds."""
        job_spec = {
            "taskGroups": [{
                "taskSpec": {
                    "maxRunDuration": "7200s",
                }
            }],
        }

        # Should pass with higher limit
        validate_max_runtime(job_spec, max_seconds=7200)


class TestEnforceGuardrails:
    """Tests for enforce_guardrails function."""

    def test_valid_job_spec(self):
        """Test enforcement passes with valid spec."""
        job_spec = {
            "allocationPolicy": {
                "instances": [{
                    "policy": {
                        "provisioningModel": "SPOT",
                    }
                }]
            },
            "taskGroups": [{
                "taskSpec": {
                    "maxRunDuration": "3600s",
                }
            }],
        }

        result = enforce_guardrails(job_spec)
        assert result == job_spec

    def test_invalid_job_spec(self):
        """Test enforcement fails with invalid spec."""
        job_spec = {
            "allocationPolicy": {},
            "taskGroups": [{}],
        }

        with pytest.raises(GuardrailViolation):
            enforce_guardrails(job_spec)


class TestCheckBucketLifecycle:
    """Tests for check_bucket_lifecycle function."""

    @patch("google.cloud.storage.Client")
    def test_bucket_with_rules(self, mock_client_class):
        """Test bucket with lifecycle rules."""
        mock_bucket = MagicMock()
        mock_bucket.lifecycle_rules = [MagicMock()]

        mock_client = MagicMock()
        mock_client.bucket.return_value = mock_bucket
        mock_client_class.return_value = mock_client

        result = check_bucket_lifecycle("test-bucket")

        assert result is True

    @patch("google.cloud.storage.Client")
    def test_bucket_without_rules(self, mock_client_class):
        """Test bucket without lifecycle rules."""
        mock_bucket = MagicMock()
        mock_bucket.lifecycle_rules = []

        mock_client = MagicMock()
        mock_client.bucket.return_value = mock_bucket
        mock_client_class.return_value = mock_client

        result = check_bucket_lifecycle("test-bucket")

        assert result is False

    @patch("google.cloud.storage.Client")
    def test_bucket_check_error(self, mock_client_class):
        """Test handling of bucket check errors."""
        mock_client = MagicMock()
        mock_client.bucket.side_effect = Exception("API Error")
        mock_client_class.return_value = mock_client

        result = check_bucket_lifecycle("test-bucket")

        assert result is False


class TestGuardrailsIntegration:
    """Integration tests for guardrails module."""

    def test_full_job_validation(self):
        """Test complete job spec validation."""
        valid_job = {
            "allocationPolicy": {
                "instances": [{
                    "policy": {
                        "provisioningModel": "SPOT",
                        "machineType": "n1-standard-4",
                    }
                }]
            },
            "taskGroups": [{
                "taskSpec": {
                    "maxRunDuration": "3600s",
                    "runnables": [{
                        "container": {
                            "imageUri": "python:3.12",
                        }
                    }],
                }
            }],
        }

        # Should pass all validations
        result = enforce_guardrails(valid_job)
        assert result is valid_job

    def test_spot_enforcement_critical(self):
        """Test that Spot enforcement is critical."""
        job_without_spot = {
            "allocationPolicy": {
                "instances": [{
                    "policy": {
                        "provisioningModel": "STANDARD",
                    }
                }]
            },
            "taskGroups": [{
                "taskSpec": {
                    "maxRunDuration": "3600s",
                }
            }],
        }

        with pytest.raises(GuardrailViolation) as exc_info:
            enforce_guardrails(job_without_spot)

        assert "CRITICAL" in str(exc_info.value)
