"""Main CLI entry point for G-Force."""

import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text

from gforce.core.auth import (
    AuthenticationError,
    check_auth_silent,
    get_auth_status_message,
    get_project_id,
    require_auth,
    validate_adc,
)
from gforce.core.batch import (
    BatchJobClient,
    JobConfig,
    create_inference_job,
    create_training_job,
    get_job_status,
    list_active_jobs,
)
from gforce.core.cache import ModelCache
from gforce.core.config import GForceConfig, get_config, set_config
from gforce.core.guardrails import (
    CostEstimate,
    display_cost_confirmation,
    get_spot_price_estimate,
)
from gforce.infra.stack import (
    deploy_infrastructure,
    destroy_infrastructure,
    get_infrastructure_status,
    PulumiStackManager,
)

app = typer.Typer(
    name="gforce",
    help="G-Force: Disposable Compute for ML workflows on GCP",
    rich_markup_mode="rich",
)

console = Console()


# Helper function for progress display
def with_progress(description: str, func, *args, **kwargs):
    """Run a function with a progress spinner."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(description=description, total=None)
        try:
            result = func(*args, **kwargs)
            progress.update(task, completed=True)
            return result
        except Exception as e:
            progress.update(task, completed=True)
            raise


@app.callback()
def main_callback(
    project: Optional[str] = typer.Option(
        None,
        "--project",
        "-p",
        help="GCP project ID (overrides environment)",
        envvar="GFORCE_GCP_PROJECT",
    ),
    region: Optional[str] = typer.Option(
        None,
        "--region",
        "-r",
        help="GCP region (overrides environment)",
        envvar="GFORCE_GCP_REGION",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose output",
    ),
) -> None:
    """G-Force CLI - Disposable Compute for ML workflows."""
    # Load config with potential overrides
    config_dict = {}
    if project:
        config_dict["gcp_project"] = project
    if region:
        config_dict["gcp_region"] = region

    if config_dict:
        # Create new config with overrides
        base_config = get_config()
        new_config = GForceConfig(
            **{
                **base_config.model_dump(),
                **config_dict,
            }
        )
        set_config(new_config)


@app.command()
def init(
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip confirmation prompts",
    ),
) -> None:
    """Initialize G-Force infrastructure and configuration."""
    console.print(Panel(
        Text.from_markup(
            "[bold blue]G-Force Initialization[/bold blue]\n\n"
            "This will:\n"
            "  1. Enable required GCP APIs\n"
            "  2. Create the GCS bucket for assets\n"
            "  3. Configure Pulumi state backend\n"
        ),
        border_style="blue",
    ))

    # Check authentication
    try:
        project_id = get_project_id()
        console.print(f"[green]✓[/green] Authenticated to project: {project_id}")
    except AuthenticationError as e:
        console.print(f"[red]✗[/red] {e}")
        console.print(
            "[yellow]Please run:[/yellow] "
            "[bold]gcloud auth application-default login[/bold]"
        )
        raise typer.Exit(1)

    config = get_config()

    # Show what will be created
    bucket_name = config.get_bucket_name()
    console.print(f"\n[bold]Configuration:[/bold]")
    console.print(f"  Project: {config.gcp_project}")
    console.print(f"  Region: {config.gcp_region}")
    console.print(f"  Bucket: {bucket_name}")
    console.print(f"  Pulumi State: {config.get_pulumi_state_url()}")

    if not yes:
        confirmed = console.input(
            "\nProceed with initialization? [y/N]: "
        )
        if confirmed.strip().lower() != "y":
            console.print("Initialization cancelled.")
            raise typer.Exit(0)

    # Enable APIs
    with console.status("[bold green]Enabling GCP APIs...") as status:
        from gforce.core.bootstrap import enable_required_apis

        try:
            enable_required_apis(project_id)
            status.update("[bold green]✓ APIs enabled")
        except Exception as e:
            console.print(f"[red]✗ Failed to enable APIs: {e}[/red]")
            raise typer.Exit(1)

    # Create bucket and configure Pulumi
    with console.status("[bold green]Deploying infrastructure...") as status:
        try:
            result = deploy_infrastructure(config=config)
            status.update("[bold green]✓ Infrastructure deployed")
        except Exception as e:
            console.print(f"[red]✗ Deployment failed: {e}[/red]")
            raise typer.Exit(1)

    console.print(f"\n[bold green]✓ G-Force initialized successfully![/bold green]")
    console.print(f"\n[bold]Resources created:[/bold]")
    console.print(f"  Bucket: gs://{result.get('bucket_name', bucket_name)}")
    console.print(f"  Service Account: {result.get('service_account_email', 'N/A')}")


@app.command()
def infra_up(
    preview: bool = typer.Option(
        False,
        "--preview",
        help="Preview changes without deploying",
    ),
) -> None:
    """Deploy or update G-Force infrastructure."""
    require_auth(lambda: None)()

    config = get_config()

    console.print(f"[bold]Deploying infrastructure to {config.gcp_project}...[/bold]")

    try:
        result = deploy_infrastructure(preview_only=preview, config=config)

        if preview:
            console.print("\n[bold]Preview complete.[/bold]")
        else:
            console.print("\n[bold green]✓ Infrastructure deployed![/bold green]")
            console.print(f"  Bucket: {result.get('bucket_url', 'N/A')}")
    except Exception as e:
        console.print(f"[red]✗ Deployment failed: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def infra_down(
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Skip confirmation",
    ),
) -> None:
    """Destroy G-Force infrastructure."""
    require_auth(lambda: None)()

    if not force:
        console.print(
            "[bold red]⚠️  WARNING:[/bold red] This will destroy all G-Force infrastructure!"
        )
        confirmed = console.input("Type 'destroy' to confirm: ")
        if confirmed != "destroy":
            console.print("Cancelled.")
            raise typer.Exit(0)

    try:
        result = destroy_infrastructure(force=force)
        if result.get("destroyed"):
            console.print("[bold green]✓ Infrastructure destroyed[/bold green]")
        else:
            console.print("[yellow]Infrastructure destruction may have failed[/yellow]")
    except Exception as e:
        console.print(f"[red]✗ Destruction failed: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def train(
    dataset: str = typer.Argument(
        ...,
        help="GCS URI to training dataset (e.g., gs://bucket/dataset)",
    ),
    output: str = typer.Option(
        ...,
        "--output",
        "-o",
        help="Output checkpoint name",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="Base model ID (uses default if not specified)",
    ),
    prompt: Optional[str] = typer.Option(
        None,
        "--prompt",
        "-p",
        help="Instance prompt (e.g., 'photo of sks person')",
    ),
    steps: int = typer.Option(
        1000,
        "--steps",
        "-s",
        help="Number of training steps",
    ),
    hf_token: Optional[str] = typer.Option(
        None,
        "--hf-token",
        help="HuggingFace token for gated models (or set HF_TOKEN env var)",
        envvar="HF_TOKEN",
    ),
    use_custom_image: bool = typer.Option(
        True,
        "--custom-image/--no-custom-image",
        help="Use pre-built worker image (faster) or install deps at runtime",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip cost confirmation",
    ),
) -> None:
    """Run DreamBooth training on a T4 GPU."""
    require_auth(lambda: None)()

    config = get_config()

    # Validate dataset path
    if not dataset.startswith("gs://"):
        console.print("[red]Error: Dataset must be a gs:// URI[/red]")
        raise typer.Exit(1)

    # Get cost estimate
    estimate = get_spot_price_estimate(config=config)

    # Display confirmation
    if not yes:
        if not display_cost_confirmation(estimate):
            console.print("Cancelled.")
            raise typer.Exit(0)

    # Create training job
    with console.status("[bold green]Creating training job..."):
        try:
            job = create_training_job(
                dataset_uri=dataset,
                output_name=output,
                model_id=model,
                instance_prompt=prompt,
                num_steps=steps,
                hf_token=hf_token,
                use_custom_image=use_custom_image,
                config=config,
            )
        except Exception as e:
            console.print(f"[red]✗ Failed to create job: {e}[/red]")
            raise typer.Exit(1)

    console.print(f"[bold green]✓ Training job created![/bold green]")
    console.print(f"\n[bold]Job Details:[/bold]")
    console.print(f"  Name: {job.name}")
    console.print(f"  Output: gs://{config.get_bucket_name()}/outputs/{output}/")
    console.print(f"\n[bold]Monitor with:[/bold] gforce status")


@app.command()
def gen(
    prompt: str = typer.Argument(
        ...,
        help="Generation prompt",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="Model path or ID",
    ),
    num_images: int = typer.Option(
        10,
        "--num-images",
        "-n",
        help="Number of images to generate",
    ),
    output_prefix: str = typer.Option(
        "gen",
        "--output",
        "-o",
        help="Output prefix for generated images",
    ),
    hf_token: Optional[str] = typer.Option(
        None,
        "--hf-token",
        help="HuggingFace token for gated models (or set HF_TOKEN env var)",
        envvar="HF_TOKEN",
    ),
    use_custom_image: bool = typer.Option(
        True,
        "--custom-image/--no-custom-image",
        help="Use pre-built worker image (faster) or install deps at runtime",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip cost confirmation",
    ),
) -> None:
    """Run batch inference on a T4 GPU."""
    require_auth(lambda: None)()

    config = get_config()

    # Get cost estimate
    estimate = get_spot_price_estimate(config=config)

    # Display confirmation
    if not yes:
        if not display_cost_confirmation(estimate):
            console.print("Cancelled.")
            raise typer.Exit(0)

    # Create inference job
    with console.status("[bold green]Creating inference job..."):
        try:
            job = create_inference_job(
                prompt=prompt,
                model_path=model,
                num_images=num_images,
                output_prefix=output_prefix,
                hf_token=hf_token,
                use_custom_image=use_custom_image,
                config=config,
            )
        except Exception as e:
            console.print(f"[red]✗ Failed to create job: {e}[/red]")
            raise typer.Exit(1)

    console.print(f"[bold green]✓ Inference job created![/bold green]")
    console.print(f"\n[bold]Job Details:[/bold]")
    console.print(f"  Name: {job.name}")
    console.print(f"  Output: gs://{config.get_bucket_name()}/outputs/{output_prefix}/")
    console.print(f"\n[bold]Monitor with:[/bold] gforce status")


@app.command()
def status(
    job_id: Optional[str] = typer.Argument(
        None,
        help="Specific job ID to check (optional)",
    ),
) -> None:
    """Check status of G-Force jobs."""
    require_auth(lambda: None)()

    config = get_config()

    if job_id:
        # Show detailed status for specific job
        try:
            status_info = get_job_status(job_id, config=config)

            table = Table(title=f"Job Status: {job_id}")
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="green")

            for key, value in status_info.items():
                table.add_row(key, str(value))

            console.print(table)
        except Exception as e:
            console.print(f"[red]✗ Failed to get job status: {e}[/red]")
            raise typer.Exit(1)
    else:
        # List all active jobs
        try:
            jobs = list_active_jobs(config=config)

            if not jobs:
                console.print("[dim]No active jobs found.[/dim]")
            else:
                table = Table(title="Active Jobs")
                table.add_column("Job Name", style="cyan")
                table.add_column("State", style="yellow")
                table.add_column("Created", style="green")

                for job in jobs:
                    table.add_row(
                        job["name"].split("/")[-1],
                        job["state"],
                        job["create_time"][:19] if job["create_time"] else "N/A",
                    )

                console.print(table)
        except Exception as e:
            console.print(f"[red]✗ Failed to list jobs: {e}[/red]")
            raise typer.Exit(1)


@app.command()
def pull(
    output_name: str = typer.Argument(
        ...,
        help="Output name or prefix to download",
    ),
    destination: Path = typer.Argument(
        Path("./outputs"),
        help="Local destination directory",
    ),
) -> None:
    """Download outputs from GCS to local directory."""
    require_auth(lambda: None)()

    config = get_config()
    bucket_name = config.get_bucket_name()
    source_path = f"gs://{bucket_name}/outputs/{output_name}/"

    console.print(f"[bold]Downloading from {source_path}...[/bold]")

    # Ensure destination exists
    destination.mkdir(parents=True, exist_ok=True)

    # Use gsutil for download
    import subprocess

    result = subprocess.run(
        ["gsutil", "-m", "cp", "-r", source_path, str(destination)],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        console.print(f"[red]✗ Download failed: {result.stderr}[/red]")
        raise typer.Exit(1)

    console.print(f"[bold green]✓ Downloaded to {destination}/{output_name}/[/bold green]")


@app.command()
def auth_status() -> None:
    """Check authentication status."""
    message = get_auth_status_message()
    console.print(Panel(message, title="Authentication Status", border_style="blue"))

    if check_auth_silent():
        try:
            project_id = get_project_id()
            console.print(f"\n[bold]Project:[/bold] {project_id}")
            console.print(f"[bold]Region:[/bold] {get_config().gcp_region}")
        except Exception:
            pass


@app.command()
def cache_list() -> None:
    """List cached models in GCS."""
    require_auth(lambda: None)()

    config = get_config()
    cache = ModelCache(config)

    # List all models in cache
    prefix = f"{config.cache_prefix}/"
    blobs = list(cache.client.list_blobs(cache.bucket, prefix=prefix))

    # Extract unique models
    models = set()
    for blob in blobs:
        parts = blob.name[len(prefix) :].split("/")
        if len(parts) >= 2:
            provider = parts[0]
            repo_id = parts[1].replace("--", "/")
            models.add((provider, repo_id))

    if not models:
        console.print("[dim]No cached models found.[/dim]")
    else:
        table = Table(title="Cached Models")
        table.add_column("Provider", style="cyan")
        table.add_column("Model", style="green")

        for provider, repo_id in sorted(models):
            table.add_row(provider, repo_id)

        console.print(table)


if __name__ == "__main__":
    app()
