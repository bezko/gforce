"""Pulumi infrastructure definitions for G-Force."""

import pulumi
import pulumi_gcp as gcp
from pulumi import automation as auto

from gforce.core.config import GForceConfig, get_config


def create_infrastructure(config: GForceConfig | None = None) -> dict:
    """Create Pulumi infrastructure resources.

    Args:
        config: GForceConfig instance

    Returns:
        Dictionary of created resources
    """
    cfg = config or get_config()
    project = cfg.gcp_project or gcp.config.project
    region = cfg.gcp_region

    if not project:
        raise ValueError("GCP project must be configured")

    resources = {}

    # Create GCS bucket for assets
    bucket_name = cfg.get_bucket_name()
    bucket = gcp.storage.Bucket(
        "gforce-assets",
        name=bucket_name,
        location=region.upper(),
        storage_class="STANDARD",
        uniform_bucket_level_access=True,
        versioning=gcp.storage.BucketVersioningArgs(
            enabled=True,
        ),
        lifecycle_rules=[
            # Delete old outputs after 30 days
            gcp.storage.BucketLifecycleRuleArgs(
                action=gcp.storage.BucketLifecycleRuleActionArgs(
                    type="Delete",
                ),
                condition=gcp.storage.BucketLifecycleRuleConditionArgs(
                    age=30,
                    matches_prefixes=["outputs/"],
                ),
            ),
            # Delete old cache after 90 days
            gcp.storage.BucketLifecycleRuleArgs(
                action=gcp.storage.BucketLifecycleRuleActionArgs(
                    type="Delete",
                ),
                condition=gcp.storage.BucketLifecycleRuleConditionArgs(
                    age=90,
                    matches_prefixes=["cache/"],
                ),
            ),
        ],
        labels={
            "managed_by": "gforce",
            "environment": cfg.pulumi_stack,
        },
    )
    resources["bucket"] = bucket

    # Create a service account for Batch jobs
    service_account = gcp.serviceaccount.Account(
        "gforce-worker",
        account_id="gforce-worker",
        display_name="G-Force Batch Worker",
        description="Service account for G-Force Batch job workers",
    )
    resources["service_account"] = service_account

    # Grant necessary roles to the service account
    storage_role = gcp.projects.IAMMember(
        "gforce-worker-storage",
        project=project,
        role="roles/storage.objectAdmin",
        member=pulumi.Output.concat("serviceAccount:", service_account.email),
    )
    resources["storage_role"] = storage_role

    batch_role = gcp.projects.IAMMember(
        "gforce-worker-batch",
        project=project,
        role="roles/batch.jobsEditor",
        member=pulumi.Output.concat("serviceAccount:", service_account.email),
    )
    resources["batch_role"] = batch_role

    logging_role = gcp.projects.IAMMember(
        "gforce-worker-logging",
        project=project,
        role="roles/logging.logWriter",
        member=pulumi.Output.concat("serviceAccount:", service_account.email),
    )
    resources["logging_role"] = logging_role

    # Create VPC network for Batch jobs
    network = gcp.compute.Network(
        "gforce-network",
        auto_create_subnetworks=False,
        description="Network for G-Force Batch jobs",
    )
    resources["network"] = network

    subnet = gcp.compute.Subnetwork(
        "gforce-subnet",
        network=network.id,
        ip_cidr_range="10.0.0.0/24",
        region=region,
        private_ip_google_access=True,
    )
    resources["subnet"] = subnet

    # Firewall rule to allow internal traffic
    firewall = gcp.compute.Firewall(
        "gforce-firewall",
        network=network.id,
        allows=[
            gcp.compute.FirewallAllowArgs(
                protocol="tcp",
                ports=["0-65535"],
            ),
        ],
        source_ranges=["10.0.0.0/24"],
    )
    resources["firewall"] = firewall

    # Export outputs
    pulumi.export("bucket_name", bucket.name)
    pulumi.export("bucket_url", pulumi.Output.concat("gs://", bucket.name))
    pulumi.export("service_account_email", service_account.email)
    pulumi.export("network_id", network.id)
    pulumi.export("subnet_id", subnet.id)

    return resources


class PulumiStackManager:
    """Manages Pulumi stacks programmatically."""

    def __init__(self, config: GForceConfig | None = None):
        self.config = config or get_config()
        self.project_name = "gforce"
        self.stack_name = self.config.pulumi_stack
        self.work_dir = "."

    def _create_or_select_stack(self) -> auto.Stack:
        """Create or select a Pulumi stack."""
        def pulumi_program():
            create_infrastructure(self.config)

        stack = auto.create_or_select_stack(
            stack_name=self.stack_name,
            project_name=self.project_name,
            program=pulumi_program,
            work_dir=self.work_dir,
        )
        return stack

    def preview(self) -> auto.PreviewResult:
        """Preview infrastructure changes."""
        stack = self._create_or_select_stack()

        # Set configuration
        stack.set_config("gcp:project", auto.ConfigValue(value=self.config.gcp_project or ""))
        stack.set_config("gcp:region", auto.ConfigValue(value=self.config.gcp_region))

        return stack.preview()

    def up(self, **kwargs) -> auto.UpResult:
        """Deploy infrastructure."""
        stack = self._create_or_select_stack()

        # Set configuration
        if self.config.gcp_project:
            stack.set_config(
                "gcp:project",
                auto.ConfigValue(value=self.config.gcp_project),
            )
        stack.set_config("gcp:region", auto.ConfigValue(value=self.config.gcp_region))

        return stack.up(**kwargs)

    def destroy(self, **kwargs) -> auto.DestroyResult:
        """Destroy infrastructure."""
        stack = self._create_or_select_stack()
        return stack.destroy(**kwargs)

    def get_outputs(self) -> dict:
        """Get stack outputs."""
        stack = self._create_or_select_stack()
        return stack.outputs()

    def refresh(self) -> auto.RefreshResult:
        """Refresh stack state."""
        stack = self._create_or_select_stack()
        return stack.refresh()


def deploy_infrastructure(
    preview_only: bool = False,
    config: GForceConfig | None = None,
) -> dict:
    """Deploy G-Force infrastructure.

    Args:
        preview_only: If True, only preview changes
        config: GForceConfig instance

    Returns:
        Stack outputs
    """
    manager = PulumiStackManager(config)

    if preview_only:
        result = manager.preview()
        return {"preview": str(result)}

    result = manager.up()
    return {
        "bucket_name": result.outputs.get("bucket_name", {}).get("value"),
        "bucket_url": result.outputs.get("bucket_url", {}).get("value"),
        "service_account_email": result.outputs.get("service_account_email", {}).get("value"),
    }


def destroy_infrastructure(
    force: bool = False,
    config: GForceConfig | None = None,
) -> dict:
    """Destroy G-Force infrastructure.

    Args:
        force: Skip confirmation
        config: GForceConfig instance

    Returns:
        Destroy result summary
    """
    manager = PulumiStackManager(config)
    result = manager.destroy()
    return {"destroyed": result.summary.result == "succeeded"}


def get_infrastructure_status(config: GForceConfig | None = None) -> dict:
    """Get current infrastructure status.

    Args:
        config: GForceConfig instance

    Returns:
        Status information
    """
    manager = PulumiStackManager(config)
    outputs = manager.get_outputs()

    return {
        "stack_name": config.pulumi_stack if config else get_config().pulumi_stack,
        "bucket_name": outputs.get("bucket_name", {}).get("value"),
        "bucket_url": outputs.get("bucket_url", {}).get("value"),
        "service_account_email": outputs.get("service_account_email", {}).get("value"),
    }
