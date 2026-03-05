"""Authentication utilities for G-Force."""

import os
from functools import wraps
from typing import Callable, TypeVar

from google.auth import default as google_auth_default
from google.auth.exceptions import DefaultCredentialsError
from google.oauth2.credentials import Credentials
from rich.console import Console

console = Console()

F = TypeVar("F", bound=Callable[..., object])


class AuthenticationError(Exception):
    """Raised when authentication fails."""

    pass


def validate_adc() -> tuple[Credentials, str | None]:
    """Validate Google Application Default Credentials.

    Returns:
        Tuple of (credentials, project_id)

    Raises:
        AuthenticationError: If credentials are not valid.
    """
    # First check if GOOGLE_APPLICATION_CREDENTIALS is set
    creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if creds_path and not os.path.exists(creds_path):
        raise AuthenticationError(
            f"GOOGLE_APPLICATION_CREDENTIALS points to non-existent file: {creds_path}"
        )

    try:
        credentials, project_id = google_auth_default()
        # Force token refresh to validate credentials
        # Use hasattr checks for credential attributes that may not exist on all types
        try:
            expired = getattr(credentials, "expired", False)
            valid = getattr(credentials, "valid", True)
            needs_refresh = expired or not valid
        except AttributeError:
            needs_refresh = False
        
        if needs_refresh and hasattr(credentials, "refresh"):
                from google.auth.transport.requests import Request

                credentials.refresh(Request())
        return credentials, project_id
    except DefaultCredentialsError as e:
        raise AuthenticationError(
            "Invalid Google Application Default Credentials. "
            "Please run: gcloud auth application-default login"
        ) from e


def get_project_id() -> str:
    """Get the GCP project ID from ADC.

    Returns:
        The GCP project ID.

    Raises:
        AuthenticationError: If project ID cannot be determined.
    """
    _, project_id = validate_adc()
    if project_id is None:
        raise AuthenticationError(
            "Could not determine GCP project ID from credentials. "
            "Please set GFORCE_GCP_PROJECT environment variable."
        )
    return project_id


def require_auth(func: F) -> F:
    """Decorator that validates ADC before running a function.

    If authentication fails, prints an error message and exits.
    """

    @wraps(func)
    def wrapper(*args: object, **kwargs: object) -> object:
        try:
            validate_adc()
        except AuthenticationError as e:
            console.print(f"[bold red]Authentication Error:[/bold red] {e}")
            console.print(
                "[yellow]Please run:[/yellow] "
                "[bold]gcloud auth application-default login[/bold]"
            )
            raise SystemExit(1) from e
        return func(*args, **kwargs)

    return wrapper  # type: ignore[return-value]


def check_auth_silent() -> bool:
    """Check if ADC is valid without printing anything.

    Returns:
        True if credentials are valid, False otherwise.
    """
    try:
        validate_adc()
        return True
    except AuthenticationError:
        return False


def get_auth_status_message() -> str:
    """Get a user-friendly authentication status message."""
    try:
        credentials, project_id = validate_adc()
        if project_id:
            return (
                f"[green]Authenticated[/green] to project "
                f"[bold]{project_id}[/bold]"
            )
        return "[green]Authenticated[/green] (project unknown)"
    except AuthenticationError as e:
        return f"[red]Not authenticated:[/red] {e}"
