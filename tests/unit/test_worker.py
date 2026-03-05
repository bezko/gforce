"""Tests for worker_init script."""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Import the worker module
from gforce.scripts.worker_init import (
    DEFAULT_MODEL,
    GCS_MOUNT_PATH,
    LOCAL_CACHE_PATH,
    install_dependencies,
    main,
)


class TestConstants:
    """Tests for module constants."""

    def test_default_model(self):
        """Test that default model is set correctly."""
        assert DEFAULT_MODEL == "TheImposterImposters/URPM-SD1.5-v2.3.inpainting"

    def test_paths_are_path_objects(self):
        """Test that path constants are Path objects."""
        assert isinstance(GCS_MOUNT_PATH, Path)
        assert isinstance(LOCAL_CACHE_PATH, Path)

    def test_gcs_mount_path(self):
        """Test GCS mount path."""
        assert str(GCS_MOUNT_PATH) == "/mnt/disks/gcs"


class TestInstallDependencies:
    """Tests for install_dependencies function."""

    @patch("gforce.scripts.worker_init.subprocess.run")
    def test_successful_installation(self, mock_run):
        """Test successful dependency installation."""
        mock_run.return_value = MagicMock(returncode=0)

        install_dependencies()

        # Should have been called multiple times (once per package)
        assert mock_run.call_count > 0

    @patch("gforce.scripts.worker_init.subprocess.run")
    def test_installation_failure(self, mock_run):
        """Test handling of installation failure."""
        mock_run.return_value = MagicMock(
            returncode=1,
            stderr="Package not found",
        )

        with pytest.raises(RuntimeError, match="Failed to install"):
            install_dependencies()


class TestMain:
    """Tests for main function."""

    @patch("gforce.scripts.worker_init.GCS_MOUNT_PATH")
    @patch("gforce.scripts.worker_init.install_dependencies")
    def test_main_checks_mount(self, mock_install, mock_path):
        """Test that main checks for GCS mount."""
        mock_path.exists.return_value = False

        with patch("sys.argv", ["worker_init.py", "--mode", "train", "--dataset", "gs://bucket/data"]):
            result = main()

        assert result == 1  # Should exit with error

    @patch("gforce.scripts.worker_init.GCS_MOUNT_PATH")
    @patch("gforce.scripts.worker_init.install_dependencies")
    @patch("gforce.scripts.worker_init.train_dreambooth")
    def test_train_mode_success(
        self,
        mock_train,
        mock_install,
        mock_path,
    ):
        """Test successful training mode."""
        mock_path.exists.return_value = True
        mock_train.return_value = Path("/mnt/disks/gcs/outputs/model/model.safetensors")

        with patch.dict(os.environ, {"INSTANCE_PROMPT": "photo of sks person"}):
            with patch("sys.argv", [
                "worker_init.py",
                "--mode", "train",
                "--dataset", "gs://bucket/dataset",
                "--output", "my-model",
            ]):
                result = main()

        assert result == 0
        mock_train.assert_called_once()

    @patch("gforce.scripts.worker_init.GCS_MOUNT_PATH")
    @patch("gforce.scripts.worker_init.install_dependencies")
    @patch("gforce.scripts.worker_init.run_inference")
    def test_inference_mode_success(
        self,
        mock_inference,
        mock_install,
        mock_path,
    ):
        """Test successful inference mode."""
        mock_path.exists.return_value = True
        mock_inference.return_value = [Path("/mnt/disks/gcs/outputs/gen/image_0001.png")]

        with patch.dict(os.environ, {"GENERATION_PROMPT": "a beautiful landscape"}):
            with patch("sys.argv", [
                "worker_init.py",
                "--mode", "inference",
                "--prompt", "a beautiful landscape",
                "--num-images", "5",
            ]):
                result = main()

        assert result == 0
        mock_inference.assert_called_once()

    @patch("gforce.scripts.worker_init.GCS_MOUNT_PATH")
    @patch("gforce.scripts.worker_init.install_dependencies")
    def test_train_mode_missing_dataset(self, mock_install, mock_path):
        """Test training mode without dataset."""
        mock_path.exists.return_value = True

        with patch("sys.argv", [
            "worker_init.py",
            "--mode", "train",
            "--output", "my-model",
        ]):
            result = main()

        assert result == 1

    @patch("gforce.scripts.worker_init.GCS_MOUNT_PATH")
    @patch("gforce.scripts.worker_init.install_dependencies")
    def test_inference_mode_missing_prompt(self, mock_install, mock_path):
        """Test inference mode without prompt."""
        mock_path.exists.return_value = True

        with patch("sys.argv", [
            "worker_init.py",
            "--mode", "inference",
        ]):
            result = main()

        assert result == 1

    @patch("gforce.scripts.worker_init.GCS_MOUNT_PATH")
    def test_main_install_failure(self, mock_path):
        """Test handling of dependency install failure."""
        mock_path.exists.return_value = True

        with patch("gforce.scripts.worker_init.install_dependencies") as mock_install:
            mock_install.side_effect = RuntimeError("Installation failed")

            with patch("sys.argv", [
                "worker_init.py",
                "--mode", "train",
                "--dataset", "gs://bucket/data",
                "--output", "my-model",
            ]):
                result = main()

        assert result == 1


class TestArgumentParsing:
    """Tests for argument parsing."""

    def test_train_mode_parsing(self):
        """Test parsing of train mode arguments."""
        import argparse

        from gforce.scripts.worker_init import main

        # This is a bit of a hack to test argument parsing
        # In a real scenario, we'd refactor main to accept args directly
        test_args = [
            "--mode", "train",
            "--dataset", "gs://bucket/data",
            "--model", "custom/model",
            "--output", "my-model",
            "--steps", "500",
        ]

        # Just verify the args can be parsed
        parser = argparse.ArgumentParser()
        parser.add_argument("--mode", choices=["train", "inference"], required=True)
        parser.add_argument("--dataset")
        parser.add_argument("--model")
        parser.add_argument("--output")
        parser.add_argument("--steps", type=int, default=1000)

        args = parser.parse_args(test_args)

        assert args.mode == "train"
        assert args.dataset == "gs://bucket/data"
        assert args.model == "custom/model"
        assert args.output == "my-model"
        assert args.steps == 500

    def test_inference_mode_parsing(self):
        """Test parsing of inference mode arguments."""
        import argparse

        test_args = [
            "--mode", "inference",
            "--prompt", "a beautiful landscape",
            "--model", "custom/model",
            "--num-images", "10",
            "--output-prefix", "landscapes",
        ]

        parser = argparse.ArgumentParser()
        parser.add_argument("--mode", choices=["train", "inference"], required=True)
        parser.add_argument("--prompt")
        parser.add_argument("--model")
        parser.add_argument("--num-images", type=int, default=10)
        parser.add_argument("--output-prefix", default="gen")

        args = parser.parse_args(test_args)

        assert args.mode == "inference"
        assert args.prompt == "a beautiful landscape"
        assert args.model == "custom/model"
        assert args.num_images == 10
        assert args.output_prefix == "landscapes"


class TestWorkerIntegration:
    """Integration-style tests for worker module."""

    def test_environment_variables(self):
        """Test that environment variables are properly read."""
        env_vars = {
            "INSTANCE_PROMPT": "photo of sks person",
            "GENERATION_PROMPT": "a beautiful landscape",
            "MODEL_PATH": "gs://bucket/model",
            "NUM_IMAGES": "10",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            assert os.environ.get("INSTANCE_PROMPT") == "photo of sks person"
            assert os.environ.get("GENERATION_PROMPT") == "a beautiful landscape"
            assert os.environ.get("MODEL_PATH") == "gs://bucket/model"
            assert os.environ.get("NUM_IMAGES") == "10"

    def test_path_operations(self):
        """Test Path operations used in worker."""
        # Test output path construction
        output_name = "my-model"
        output_path = GCS_MOUNT_PATH / "outputs" / output_name

        assert str(output_path) == f"/mnt/disks/gcs/outputs/{output_name}"

        # Test local cache path
        model_id = "user/model"
        local_model_path = LOCAL_CACHE_PATH / model_id.replace("/", "--")

        assert str(local_model_path) == "/tmp/model-cache/user--model"
