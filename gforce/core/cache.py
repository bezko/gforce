"""Model caching utilities for G-Force."""

import hashlib
import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from google.cloud import storage
from rich.console import Console

from gforce.core.config import GForceConfig, get_config

logger = logging.getLogger(__name__)
console = Console()


@dataclass
class CacheEntry:
    """Represents a cached model entry."""

    provider: str
    repo_id: str
    commit_hash: str
    local_path: str | None = None
    gcs_path: str | None = None

    def get_cache_key(self) -> str:
        """Generate a unique cache key for this entry."""
        key_data = f"{self.provider}/{self.repo_id}/{self.commit_hash}"
        return hashlib.sha256(key_data.encode()).hexdigest()[:16]

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "CacheEntry":
        """Create from dictionary."""
        return cls(**data)


class ModelCache:
    """Manages model caching in GCS."""

    MANIFEST_FILE = "manifest.json"

    def __init__(self, config: GForceConfig | None = None):
        self.config = config or get_config()
        self.client = storage.Client()
        self.bucket = self.client.bucket(self.config.get_bucket_name())

    def _get_cache_path(self, entry: CacheEntry) -> str:
        """Get the GCS path for a cache entry."""
        safe_repo_id = entry.repo_id.replace("/", "--")
        return (
            f"{self.config.cache_prefix}/"
            f"{entry.provider}/"
            f"{safe_repo_id}/"
            f"{entry.commit_hash}"
        )

    def _get_manifest_path(self, entry: CacheEntry) -> str:
        """Get the GCS path for a cache manifest."""
        return f"{self._get_cache_path(entry)}/{self.MANIFEST_FILE}"

    def exists(self, entry: CacheEntry) -> bool:
        """Check if a model is already cached.

        Args:
            entry: CacheEntry to check

        Returns:
            True if the model is cached
        """
        manifest_blob = self.bucket.blob(self._get_manifest_path(entry))
        return manifest_blob.exists()

    def get_entry(self, repo_id: str, provider: str = "huggingface") -> CacheEntry | None:
        """Get cache entry for a repo if it exists.

        Args:
            repo_id: HuggingFace repo ID
            provider: Model provider

        Returns:
            CacheEntry if found, None otherwise
        """
        safe_repo_id = repo_id.replace("/", "--")
        prefix = f"{self.config.cache_prefix}/{provider}/{safe_repo_id}/"

        blobs = list(self.client.list_blobs(self.bucket, prefix=prefix))
        for blob in blobs:
            if blob.name.endswith(self.MANIFEST_FILE):
                try:
                    manifest = json.loads(blob.download_as_text())
                    return CacheEntry.from_dict(manifest)
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Invalid manifest at {blob.name}: {e}")
                    continue
        return None

    def save_manifest(self, entry: CacheEntry) -> None:
        """Save a cache manifest to GCS.

        Args:
            entry: CacheEntry to save
        """
        entry.gcs_path = self._get_cache_path(entry)
        manifest_path = self._get_manifest_path(entry)
        blob = self.bucket.blob(manifest_path)
        blob.upload_from_string(
            json.dumps(entry.to_dict(), indent=2),
            content_type="application/json",
        )
        logger.info(f"Saved cache manifest to gs://{self.bucket.name}/{manifest_path}")

    def sync_to_local(self, entry: CacheEntry, local_dir: Path | str) -> Path:
        """Sync a cached model from GCS to local disk.

        Args:
            entry: CacheEntry to sync
            local_dir: Local directory to sync to

        Returns:
            Path to the local model directory
        """
        local_path = Path(local_dir) / entry.get_cache_key()
        local_path.mkdir(parents=True, exist_ok=True)

        cache_path = self._get_cache_path(entry)
        blobs = list(self.client.list_blobs(self.bucket, prefix=cache_path))

        console.print(f"[blue]Syncing model from GCS cache...[/blue]")
        synced_count = 0

        for blob in blobs:
            # Skip manifest file
            if blob.name.endswith(self.MANIFEST_FILE):
                continue

            relative_path = blob.name[len(cache_path) + 1 :]
            local_file = local_path / relative_path
            local_file.parent.mkdir(parents=True, exist_ok=True)

            # Check if file needs updating
            if local_file.exists():
                local_hash = hashlib.md5(local_file.read_bytes()).hexdigest()
                # GCS doesn't store MD5 for composite objects, so we check size
                if local_file.stat().st_size == blob.size:
                    continue

            blob.download_to_filename(local_file)
            synced_count += 1

        console.print(f"[green]✓ Synced {synced_count} files to {local_path}[/green]")
        entry.local_path = str(local_path)
        return local_path

    def sync_to_gcs(self, entry: CacheEntry, local_path: Path | str) -> str:
        """Sync a local model to GCS cache.

        Args:
            entry: CacheEntry to create
            local_path: Local path to the model

        Returns:
            GCS path where the model was cached
        """
        local_path = Path(local_path)
        cache_path = self._get_cache_path(entry)

        console.print(f"[blue]Caching model to GCS...[/blue]")
        uploaded_count = 0

        for local_file in local_path.rglob("*"):
            if local_file.is_file():
                relative_path = local_file.relative_to(local_path)
                blob_path = f"{cache_path}/{relative_path}"
                blob = self.bucket.blob(blob_path)

                # Check if already exists (get_blob returns None if not found)
                existing = self.bucket.get_blob(blob_path)
                if existing and existing.size == local_file.stat().st_size:
                    continue

                blob.upload_from_filename(local_file)
                uploaded_count += 1

        # Save manifest
        self.save_manifest(entry)

        gcs_uri = f"gs://{self.bucket.name}/{cache_path}"
        console.print(f"[green]✓ Cached {uploaded_count} files to {gcs_uri}[/green]")
        return gcs_uri

    def get_or_download_model(
        self,
        repo_id: str,
        local_cache_dir: Path | str,
        provider: str = "huggingface",
    ) -> tuple[Path, bool]:
        """Get a model from cache or download it.

        Args:
            repo_id: HuggingFace repo ID
            local_cache_dir: Local directory for caching
            provider: Model provider

        Returns:
            Tuple of (local_path, was_cached)
        """
        # Check if we have this model cached
        existing_entry = self.get_entry(repo_id, provider)

        if existing_entry:
            console.print(f"[green]Found cached model: {repo_id}[/green]")
            local_path = self.sync_to_local(existing_entry, local_cache_dir)
            return local_path, True

        console.print(f"[yellow]Model not in GCS cache: {repo_id}[/yellow]")
        return Path(local_cache_dir), False

    def cache_local_model(
        self,
        repo_id: str,
        commit_hash: str,
        local_path: Path | str,
        provider: str = "huggingface",
    ) -> str:
        """Cache a locally downloaded model to GCS.

        Args:
            repo_id: HuggingFace repo ID
            commit_hash: Commit hash of the model version
            local_path: Local path to the model
            provider: Model provider

        Returns:
            GCS URI of the cached model
        """
        entry = CacheEntry(
            provider=provider,
            repo_id=repo_id,
            commit_hash=commit_hash,
        )
        return self.sync_to_gcs(entry, local_path)


def get_cache() -> ModelCache:
    """Get a ModelCache instance."""
    return ModelCache()
