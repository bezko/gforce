"""Cost estimation and guardrail utilities for G-Force."""

import logging
from dataclasses import dataclass

from google.cloud import billing
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from gforce.core.config import GForceConfig, get_config

logger = logging.getLogger(__name__)
console = Console()


@dataclass
class CostEstimate:
    """Cost estimate for a job run."""

    hourly_rate: float
    max_duration_hours: float
    estimated_max_cost: float
    is_spot: bool
    currency: str = "USD"

    def format_summary(self) -> str:
        """Format a user-friendly cost summary."""
        instance_type = "Spot Preemptible" if self.is_spot else "On-Demand"
        return (
            f"Instance: {instance_type}\n"
            f"Rate: ${self.hourly_rate:.2f}/hour\n"
            f"Max Duration: {self.max_duration_hours:.1f} hours\n"
            f"Estimated Max Cost: ${self.estimated_max_cost:.2f} {self.currency}"
        )


class GuardrailViolation(Exception):
    """Raised when a guardrail is violated."""

    pass


def get_spot_price_estimate(
    region: str | None = None,
    config: GForceConfig | None = None,
) -> CostEstimate:
    """Get cost estimate for a Spot T4 job.

    In a production environment, this would query the Cloud Billing API
    for real-time Spot prices. For now, we use a conservative estimate.

    Args:
        region: GCP region (defaults to config)
        config: GForceConfig instance (uses global if not provided)

    Returns:
        CostEstimate for the job
    """
    cfg = config or get_config()
    region = region or cfg.gcp_region

    # Try to get real pricing, fall back to estimate
    hourly_rate = cfg.spot_price_estimate

    max_duration_hours = cfg.max_run_duration / 3600
    estimated_max_cost = hourly_rate * max_duration_hours

    return CostEstimate(
        hourly_rate=hourly_rate,
        max_duration_hours=max_duration_hours,
        estimated_max_cost=estimated_max_cost,
        is_spot=True,
    )


def fetch_real_spot_price(
    region: str,
    machine_type: str = "n1-standard-4",
    gpu_type: str = "nvidia-tesla-t4",
) -> float | None:
    """Fetch real Spot price from Cloud Billing API.

    Args:
        region: GCP region
        machine_type: Machine type
        gpu_type: GPU type

    Returns:
        Spot price per hour, or None if unavailable
    """
    try:
        # Note: GCP Billing API doesn't directly expose Spot prices
        # This would require using the Cloud Billing Catalog API
        # For now, return None to use the fallback estimate
        return None
    except Exception as e:
        logger.debug(f"Failed to fetch real spot price: {e}")
        return None


def display_cost_confirmation(estimate: CostEstimate) -> bool:
    """Display cost estimate and prompt for confirmation.

    Args:
        estimate: CostEstimate to display

    Returns:
        True if user confirms, False otherwise
    """
    panel = Panel(
        Text.from_markup(
            f"[bold yellow]⚠️  Cost Warning[/bold yellow]\n\n"
            f"{estimate.format_summary()}\n\n"
            f"[bold]This job will automatically terminate after "
            f"{estimate.max_duration_hours:.1f} hours.[/bold]"
        ),
        title="Disposable Compute Cost Estimate",
        border_style="yellow",
    )
    console.print(panel)

    response = console.input(
        f"\nEstimated cost for this {estimate.max_duration_hours:.1f}hr run: "
        f"${estimate.estimated_max_cost:.2f}. Proceed? [y/N]: "
    )

    return response.strip().lower() == "y"


def validate_spot_configuration(job_spec: dict) -> None:
    """Validate that a Batch job spec enforces Spot provisioning.

    Args:
        job_spec: Cloud Batch job specification

    Raises:
        GuardrailViolation: If Spot is not configured
    """
    allocations = job_spec.get("taskGroups", [{}])[0].get(
        "taskSpec", {}
    ).get("computeResource", {}).get("machineType", "")

    # Check provisioning model in policy
    policy = job_spec.get("allocationPolicy", {})
    instances = policy.get("instances", [{}])[0]
    provisioning_model = instances.get("policy", {}).get("provisioningModel", "")

    if provisioning_model != "SPOT":
        raise GuardrailViolation(
            f"CRITICAL: Job does not use SPOT provisioning. "
            f"Found: {provisioning_model or 'NOT SET'}"
        )

    logger.info("✓ Spot provisioning validated")


def validate_max_runtime(job_spec: dict, max_seconds: int = 3600) -> None:
    """Validate that a Batch job has a max runtime set.

    Args:
        job_spec: Cloud Batch job specification
        max_seconds: Expected max runtime in seconds

    Raises:
        GuardrailViolation: If max runtime is not set correctly
    """
    task_spec = job_spec.get("taskGroups", [{}])[0].get("taskSpec", {})
    max_duration = task_spec.get("maxRunDuration", "")

    if not max_duration:
        raise GuardrailViolation("CRITICAL: Job does not have maxRunDuration set")

    # Parse duration (format: "3600s")
    try:
        duration_seconds = int(max_duration.rstrip("s"))
        if duration_seconds > max_seconds:
            raise GuardrailViolation(
                f"CRITICAL: Job maxRunDuration ({duration_seconds}s) "
                f"exceeds limit ({max_seconds}s)"
            )
    except ValueError as e:
        raise GuardrailViolation(
            f"CRITICAL: Invalid maxRunDuration format: {max_duration}"
        ) from e

    logger.info(f"✓ Max runtime validated ({duration_seconds}s)")


def enforce_guardrails(job_spec: dict) -> dict:
    """Enforce all guardrails on a job spec.

    Args:
        job_spec: Cloud Batch job specification

    Returns:
        The validated job spec

    Raises:
        GuardrailViolation: If any guardrail is violated
    """
    validate_spot_configuration(job_spec)
    validate_max_runtime(job_spec)
    return job_spec


def check_bucket_lifecycle(bucket_name: str) -> bool:
    """Check if a bucket has appropriate lifecycle policies.

    Args:
        bucket_name: Name of the GCS bucket

    Returns:
        True if lifecycle policies are appropriate
    """
    try:
        from google.cloud import storage
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        bucket.reload()

        rules = list(bucket.lifecycle_rules)
        if not rules:
            logger.warning(
                f"Bucket {bucket_name} has no lifecycle rules. "
                "Consider adding auto-deletion for old outputs."
            )
            return False
        return True
    except Exception as e:
        logger.warning(f"Could not check bucket lifecycle: {e}")
        return False
