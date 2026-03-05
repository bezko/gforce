"""Configuration management for G-Force."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class GForceConfig(BaseSettings):
    """G-Force configuration settings."""

    model_config = SettingsConfigDict(
        env_prefix="GFORCE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # GCP Settings
    gcp_project: str | None = Field(
        default=None,
        description="GCP project ID",
    )
    gcp_region: str = Field(
        default="us-central1",
        description="GCP region for resources",
    )
    gcp_zone: str = Field(
        default="us-central1-a",
        description="GCP zone for resources",
    )

    # Storage Settings
    bucket_name: str | None = Field(
        default=None,
        description="GCS bucket name for assets",
    )
    cache_prefix: str = Field(
        default="cache/models",
        description="Prefix for model cache in GCS",
    )
    outputs_prefix: str = Field(
        default="outputs",
        description="Prefix for training outputs in GCS",
    )

    # Compute Settings
    machine_type: str = Field(
        default="n1-standard-4",
        description="Machine type for Batch jobs",
    )
    gpu_type: str = Field(
        default="nvidia-tesla-t4",
        description="GPU type for training",
    )
    gpu_count: int = Field(
        default=1,
        description="Number of GPUs per job",
    )

    # Cost Settings
    spot_price_estimate: float = Field(
        default=0.15,
        description="Estimated hourly cost for Spot T4 instance",
    )
    max_run_duration: int = Field(
        default=3600,
        description="Maximum job runtime in seconds (1 hour)",
    )

    # Model Settings
    default_model: str = Field(
        default="TheImposterImposters/URPM-SD1.5-v2.3.inpainting",
        description="Default HuggingFace model for training",
    )

    # Pulumi Settings
    pulumi_stack: str = Field(
        default="dev",
        description="Pulumi stack name",
    )

    def get_bucket_name(self) -> str:
        """Get the bucket name, computing it if necessary."""
        if self.bucket_name:
            return self.bucket_name
        if self.gcp_project:
            return f"gforce-assets-{self.gcp_project}"
        raise ValueError(
            "Either GFORCE_BUCKET_NAME or GFORCE_GCP_PROJECT must be set"
        )

    def get_pulumi_state_url(self) -> str:
        """Get the Pulumi state backend URL."""
        return f"gs://{self.get_bucket_name()}/state"


# Global config instance
_config: GForceConfig | None = None


def get_config() -> GForceConfig:
    """Get or create the global config instance."""
    global _config
    if _config is None:
        _config = GForceConfig()
    return _config


def set_config(config: GForceConfig) -> None:
    """Set the global config instance (useful for testing)."""
    global _config
    _config = config
