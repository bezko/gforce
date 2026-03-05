"""Tests for cache module."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gforce.core.cache import CacheEntry, ModelCache
from gforce.core.config import GForceConfig


class TestCacheEntry:
    """Tests for CacheEntry dataclass."""

    def test_cache_key_generation(self):
        """Test cache key generation."""
        entry = CacheEntry(
            provider="huggingface",
            repo_id="user/model",
            commit_hash="abc123def456",
        )

        key = entry.get_cache_key()

        # Key should be deterministic
        assert len(key) == 16  # First 16 chars of SHA256
        assert key == entry.get_cache_key()  # Same result on second call

    def test_to_dict(self):
        """Test conversion to dictionary."""
        entry = CacheEntry(
            provider="huggingface",
            repo_id="user/model",
            commit_hash="abc123",
            local_path="/tmp/model",
            gcs_path="gs://bucket/cache/model",
        )

        data = entry.to_dict()

        assert data["provider"] == "huggingface"
        assert data["repo_id"] == "user/model"
        assert data["commit_hash"] == "abc123"
        assert data["local_path"] == "/tmp/model"
        assert data["gcs_path"] == "gs://bucket/cache/model"

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "provider": "huggingface",
            "repo_id": "user/model",
            "commit_hash": "abc123",
            "local_path": "/tmp/model",
            "gcs_path": "gs://bucket/cache/model",
        }

        entry = CacheEntry.from_dict(data)

        assert entry.provider == "huggingface"
        assert entry.repo_id == "user/model"
        assert entry.commit_hash == "abc123"


class TestModelCache:
    """Tests for ModelCache class."""

    @patch("gforce.core.cache.storage.Client")
    def test_init_with_default_config(self, mock_client_class):
        """Test initialization with default config."""
        config = GForceConfig(gcp_project="test-project")

        cache = ModelCache(config)

        assert cache.config == config

    @patch("gforce.core.cache.storage.Client")
    def test_get_cache_path(self, mock_client_class):
        """Test cache path generation."""
        config = GForceConfig(gcp_project="test-project")
        cache = ModelCache(config)

        entry = CacheEntry(
            provider="huggingface",
            repo_id="user/model",
            commit_hash="abc123",
        )

        path = cache._get_cache_path(entry)

        assert "cache/models/huggingface/user--model/abc123" in path

    @patch("gforce.core.cache.storage.Client")
    def test_exists_returns_true_when_manifest_exists(self, mock_client_class):
        """Test exists returns True when manifest exists."""
        mock_blob = MagicMock()
        mock_blob.exists.return_value = True

        mock_bucket = MagicMock()
        mock_bucket.blob.return_value = mock_blob

        mock_client = MagicMock()
        mock_client.bucket.return_value = mock_bucket
        mock_client_class.return_value = mock_client

        config = GForceConfig(gcp_project="test-project")
        cache = ModelCache(config)

        entry = CacheEntry(
            provider="huggingface",
            repo_id="user/model",
            commit_hash="abc123",
        )

        assert cache.exists(entry) is True

    @patch("gforce.core.cache.storage.Client")
    def test_exists_returns_false_when_manifest_missing(self, mock_client_class):
        """Test exists returns False when manifest doesn't exist."""
        mock_blob = MagicMock()
        mock_blob.exists.return_value = False

        mock_bucket = MagicMock()
        mock_bucket.blob.return_value = mock_blob

        mock_client = MagicMock()
        mock_client.bucket.return_value = mock_bucket
        mock_client_class.return_value = mock_client

        config = GForceConfig(gcp_project="test-project")
        cache = ModelCache(config)

        entry = CacheEntry(
            provider="huggingface",
            repo_id="user/model",
            commit_hash="abc123",
        )

        assert cache.exists(entry) is False

    @patch("gforce.core.cache.storage.Client")
    def test_save_manifest(self, mock_client_class):
        """Test manifest saving."""
        mock_blob = MagicMock()

        mock_bucket = MagicMock()
        mock_bucket.blob.return_value = mock_blob

        mock_client = MagicMock()
        mock_client.bucket.return_value = mock_bucket
        mock_client_class.return_value = mock_client

        config = GForceConfig(gcp_project="test-project")
        cache = ModelCache(config)

        entry = CacheEntry(
            provider="huggingface",
            repo_id="user/model",
            commit_hash="abc123",
        )

        cache.save_manifest(entry)

        mock_bucket.blob.assert_called_once()
        mock_blob.upload_from_string.assert_called_once()

        # Check that JSON was uploaded
        call_args = mock_blob.upload_from_string.call_args
        json_data = json.loads(call_args[0][0])
        assert json_data["provider"] == "huggingface"
        assert json_data["repo_id"] == "user/model"


class TestModelCacheGetEntry:
    """Tests for ModelCache.get_entry method."""

    @patch("gforce.core.cache.storage.Client")
    def test_get_entry_found(self, mock_client_class):
        """Test getting existing cache entry."""
        mock_blob = MagicMock()
        mock_blob.name = "cache/models/huggingface/user--model/abc123/manifest.json"
        mock_blob.download_as_text.return_value = json.dumps({
            "provider": "huggingface",
            "repo_id": "user/model",
            "commit_hash": "abc123",
        })

        mock_client = MagicMock()
        mock_client.list_blobs.return_value = [mock_blob]
        mock_client_class.return_value = mock_client

        config = GForceConfig(gcp_project="test-project")
        cache = ModelCache(config)

        entry = cache.get_entry("user/model")

        assert entry is not None
        assert entry.provider == "huggingface"
        assert entry.repo_id == "user/model"

    @patch("gforce.core.cache.storage.Client")
    def test_get_entry_not_found(self, mock_client_class):
        """Test getting non-existent cache entry."""
        mock_client = MagicMock()
        mock_client.list_blobs.return_value = []
        mock_client_class.return_value = mock_client

        config = GForceConfig(gcp_project="test-project")
        cache = ModelCache(config)

        entry = cache.get_entry("user/model")

        assert entry is None


class TestModelCacheSync:
    """Tests for sync methods."""

    @patch("gforce.core.cache.storage.Client")
    def test_sync_to_local_creates_directory(self, mock_client_class, tmp_path):
        """Test that sync_to_local creates the target directory."""
        mock_blob = MagicMock()
        mock_blob.name = "cache/models/huggingface/user--model/abc123/model.safetensors"
        mock_blob.size = 100

        mock_bucket = MagicMock()
        mock_bucket.get_blob.return_value = None  # File doesn't exist locally

        mock_client = MagicMock()
        mock_client.list_blobs.return_value = [mock_blob]
        mock_client.bucket.return_value = mock_bucket
        mock_client_class.return_value = mock_client

        config = GForceConfig(gcp_project="test-project")
        cache = ModelCache(config)

        entry = CacheEntry(
            provider="huggingface",
            repo_id="user/model",
            commit_hash="abc123",
        )

        local_dir = tmp_path / "models"
        result = cache.sync_to_local(entry, local_dir)

        assert result.exists()


class TestGetCache:
    """Tests for get_cache function."""

    @patch("gforce.core.cache.storage.Client")
    def test_get_cache_returns_instance(self, mock_client_class):
        """Test that get_cache returns a ModelCache instance."""
        from gforce.core.cache import get_cache

        cache = get_cache()
        assert isinstance(cache, ModelCache)


class TestCacheIntegration:
    """Integration-style tests for cache module."""

    def test_cache_entry_round_trip(self):
        """Test cache entry serialization round-trip."""
        original = CacheEntry(
            provider="huggingface",
            repo_id="stabilityai/stable-diffusion-xl-base-1.0",
            commit_hash="a1b2c3d4e5f6",
            local_path="/tmp/models/sdxl",
            gcs_path="gs://bucket/cache/models/huggingface/stabilityai--stable-diffusion-xl-base-1.0/a1b2c3d4e5f6",
        )

        # Serialize
        data = original.to_dict()

        # Deserialize
        restored = CacheEntry.from_dict(data)

        assert restored.provider == original.provider
        assert restored.repo_id == original.repo_id
        assert restored.commit_hash == original.commit_hash
        assert restored.local_path == original.local_path
        assert restored.gcs_path == original.gcs_path
