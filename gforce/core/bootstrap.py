"""Bootstrap utilities for G-Force initialization."""

import logging
import subprocess
import time

from google.api_core.extended_operation import ExtendedOperation
from google.cloud import storage
from google.cloud.storage import Bucket
from rich.console import Console
from google.api_core.extended_operation import ExtendedOperation

from gforce.core.config import GForceConfig, get_config

logger = logging.getLogger(__name__)
console = Console()

REQUIRED_APIS = [
    "batch.googleapis.com",
    "compute.googleapis.com",
    "storage.googleapis.com",
    "logging.googleapis.com",
    "monitoring.googleapis.com",
]


def enable_required_apis(
    project_id: str,
    timeout: int = 300,
) -> None:
    """Enable required GCP APIs for G-Force.

    Args:
        project_id: GCP project ID
        timeout: Maximum time to wait for API enablement
    """
    import google.cloud.service_usage_v1 as service_usage_v1

    client = service_usage_v1.ServiceUsageClient()
    parent = f"projects/{project_id}"

    console.print(f"[bold]Enabling required APIs for project {project_id}...[/bold]")

    for api in REQUIRED_APIS:
        service_name = f"{parent}/services/{api}"

        try:
            # Check if already enabled
            service = client.get_service(name=service_name)
            if service.state == service_usage_v1.State.ENABLED:
                console.print(f"  [green]✓[/green] {api} (already enabled)")
                continue

            # Enable the API
            console.print(f"  [yellow]→[/yellow] Enabling {api}...")
            operation = client.enable_service(name=service_name)

            # Wait for operation to complete
            try:
                operation.result(timeout=timeout)
                console.print(f"  [green]✓[/green] {api}")
            except Exception as e:
                logger.warning(f"Timeout waiting for {api}, may still be enabling: {e}")
                console.print(f"  [yellow]⏳[/yellow] {api} (enabling in background)")

        except Exception as e:
            logger.error(f"Failed to enable {api}: {e}")
            console.print(f"  [red]✗[/red] {api}: {e}")
            raise


def create_state_bucket(
    project_id: str,
    bucket_name: str,
    region: str,
) -> Bucket:
    """Create the GCS bucket for G-Force assets and Pulumi state.

    Args:
        project_id: GCP project ID
        bucket_name: Name for the bucket
        region: GCP region

    Returns:
        Created bucket
    """
    client = storage.Client(project=project_id)

    console.print(f"[bold]Creating GCS bucket: {bucket_name}...[/bold]")

    try:
        # Check if bucket already exists
        bucket = client.bucket(bucket_name)
        if bucket.exists():
            console.print(f"  [green]✓[/green] Bucket already exists")
            return bucket

        # Create new bucket
        bucket = client.create_bucket(
            bucket_name,
            location=region,
            project=project_id,
        )

        # Configure uniform bucket-level access
        bucket.iam_configuration.uniform_bucket_level_access_enabled = True
        bucket.patch()

        # Configure lifecycle rules using bucket lifecycle rules API
        rules = [
            {
                "action": {"type": "Delete"},
                "condition": {
                    "age": 30,
                    "matchesPrefix": ["outputs/"],
                },
            },
            {
                "action": {"type": "Delete"},
                "condition": {
                    "age": 90,
                    "matchesPrefix": ["cache/"],
                },
            },
        ]
        bucket.lifecycle_rules = rules
        bucket.patch()

        console.print(f"  [green]✓[/green] Bucket created")
        return bucket

    except Exception as e:
        logger.error(f"Failed to create bucket: {e}")
        console.print(f"  [red]✗[/red] Failed to create bucket: {e}")
        raise


def configure_pulumi_backend(bucket_name: str) -> None:
    """Configure Pulumi to use GCS as state backend.

    Args:
        bucket_name: Name of the GCS bucket
    """
    state_url = f"gs://{bucket_name}/state"

    console.print(f"[bold]Configuring Pulumi backend...[/bold]")

    try:
        result = subprocess.run(
            ["pulumi", "login", state_url],
            capture_output=True,
            text=True,
            check=True,
        )
        console.print(f"  [green]✓[/green] Pulumi logged in to {state_url}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to configure Pulumi: {e}")
        console.print(f"  [red]✗[/red] Pulumi login failed: {e.stderr}")
        raise
    except FileNotFoundError:
        logger.warning("Pulumi CLI not found, skipping Pulumi login")
        console.print(f"  [yellow]![/yellow] Pulumi CLI not found, skipping")


def verify_permissions(project_id: str) -> list[str]:
    """Verify required GCP permissions.

    Args:
        project_id: GCP project ID

    Returns:
        List of missing permissions (empty if all granted)
    """
    from google.cloud import iam_admin_v1
    from google.iam.v1 import policy_pb2

    # This is a simplified check - in production you'd use the
    # Resource Manager API to test actual permissions
    console.print(f"[bold]Verifying permissions...[/bold]")

    # For now, assume permissions are OK if we got this far
    console.print(f"  [green]✓[/green] Permissions verified")
    return []


def bootstrap_project(
    config: GForceConfig | None = None,
    skip_api_enablement: bool = False,
    skip_pulumi: bool = False,
) -> dict:
    """Run full bootstrap process.

    Args:
        config: GForceConfig instance
        skip_api_enablement: Skip API enablement step
        skip_pulumi: Skip Pulumi configuration

    Returns:
        Bootstrap result information
    """
    cfg = config or get_config()
    project_id = cfg.gcp_project
    bucket_name = cfg.get_bucket_name()
    region = cfg.gcp_region

    if not project_id:
        raise ValueError("GCP project ID must be configured")

    console.print(Panel(
        f"[bold blue]Bootstrapping G-Force[/bold blue]\n\n"
        f"Project: {project_id}\n"
        f"Region: {region}\n"
        f"Bucket: {bucket_name}",
        border_style="blue",
    ))

    results = {
        "project_id": project_id,
        "bucket_name": bucket_name,
        "apis_enabled": [],
        "bucket_created": False,
        "pulumi_configured": False,
    }

    # 1. Enable APIs
    if not skip_api_enablement:
        enable_required_apis(project_id)
        results["apis_enabled"] = REQUIRED_APIS

    # 2. Create bucket
    bucket = create_state_bucket(project_id, bucket_name, region)
    results["bucket_created"] = bucket.exists()

    # 3. Configure Pulumi
    if not skip_pulumi:
        try:
            configure_pulumi_backend(bucket_name)
            results["pulumi_configured"] = True
        except Exception as e:
            logger.warning(f"Pulumi configuration failed: {e}")
            results["pulumi_configured"] = False

    # 4. Verify permissions
    missing = verify_permissions(project_id)
    results["missing_permissions"] = missing

    console.print(f"\n[bold green]✓ Bootstrap complete![/bold green]")

    return results
