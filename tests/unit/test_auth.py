"""Tests for authentication module."""

import os
from unittest.mock import MagicMock, patch

import pytest

from gforce.core.auth import (
    AuthenticationError,
    check_auth_silent,
    get_auth_status_message,
    get_project_id,
    require_auth,
    validate_adc,
)


class TestValidateADC:
    """Tests for validate_adc function."""

    @patch("gforce.core.auth.google_auth_default")
    def test_valid_credentials(self, mock_default):
        """Test validation with valid credentials."""
        mock_creds = MagicMock()
        mock_creds.expired = False
        mock_creds.valid = True
        mock_default.return_value = (mock_creds, "test-project-123")

        creds, project = validate_adc()

        assert creds == mock_creds
        assert project == "test-project-123"

    @patch("gforce.core.auth.google_auth_default")
    @patch("google.auth.transport.requests.Request")
    def test_expired_credentials_refresh(self, mock_request_class, mock_default):
        """Test that expired credentials are refreshed."""
        mock_creds = MagicMock()
        mock_creds.expired = True
        mock_creds.valid = False
        mock_creds.refresh = MagicMock()
        mock_default.return_value = (mock_creds, "test-project")

        validate_adc()

        mock_creds.refresh.assert_called_once()

    @patch("gforce.core.auth.google_auth_default")
    def test_invalid_credentials(self, mock_default):
        """Test validation with invalid credentials."""
        from google.auth.exceptions import DefaultCredentialsError

        mock_default.side_effect = DefaultCredentialsError("No credentials found")

        with pytest.raises(AuthenticationError) as exc_info:
            validate_adc()

        assert "Invalid Google Application Default Credentials" in str(exc_info.value)
        assert "gcloud auth application-default login" in str(exc_info.value)

    @patch.dict(os.environ, {"GOOGLE_APPLICATION_CREDENTIALS": "/fake/path.json"})
    @patch("gforce.core.auth.os.path.exists")
    def test_nonexistent_creds_file(self, mock_exists):
        """Test validation when creds file doesn't exist."""
        mock_exists.return_value = False

        with pytest.raises(AuthenticationError) as exc_info:
            validate_adc()

        assert "non-existent file" in str(exc_info.value)


class TestGetProjectId:
    """Tests for get_project_id function."""

    @patch("gforce.core.auth.google_auth_default")
    def test_valid_project_id(self, mock_default):
        """Test getting project ID with valid credentials."""
        mock_creds = MagicMock()
        mock_default.return_value = (mock_creds, "my-project-id")

        project = get_project_id()
        assert project == "my-project-id"

    @patch("gforce.core.auth.google_auth_default")
    def test_missing_project_id(self, mock_default):
        """Test getting project ID when it's None."""
        mock_creds = MagicMock()
        mock_default.return_value = (mock_creds, None)

        with pytest.raises(AuthenticationError) as exc_info:
            get_project_id()

        assert "Could not determine GCP project ID" in str(exc_info.value)


class TestRequireAuth:
    """Tests for require_auth decorator."""

    @patch("gforce.core.auth.validate_adc")
    def test_decorator_passes_with_valid_auth(self, mock_validate):
        """Test that decorated function runs with valid auth."""

        @require_auth
        def test_func():
            return "success"

        result = test_func()
        assert result == "success"
        mock_validate.assert_called_once()

    @patch("gforce.core.auth.validate_adc")
    def test_decorator_exits_on_invalid_auth(self, mock_validate):
        """Test that decorated function exits with invalid auth."""
        mock_validate.side_effect = AuthenticationError("Invalid")

        @require_auth
        def test_func():
            return "success"

        with pytest.raises(SystemExit) as exc_info:
            test_func()

        assert exc_info.value.code == 1


class TestCheckAuthSilent:
    """Tests for check_auth_silent function."""

    @patch("gforce.core.auth.validate_adc")
    def test_returns_true_when_valid(self, mock_validate):
        """Test that function returns True with valid auth."""
        mock_validate.return_value = (MagicMock(), "project")

        assert check_auth_silent() is True

    @patch("gforce.core.auth.validate_adc")
    def test_returns_false_when_invalid(self, mock_validate):
        """Test that function returns False with invalid auth."""
        mock_validate.side_effect = AuthenticationError("Invalid")

        assert check_auth_silent() is False


class TestGetAuthStatusMessage:
    """Tests for get_auth_status_message function."""

    @patch("gforce.core.auth.google_auth_default")
    def test_authorized_message(self, mock_default):
        """Test message when authenticated."""
        mock_creds = MagicMock()
        mock_default.return_value = (mock_creds, "test-project")

        message = get_auth_status_message()

        assert "Authenticated" in message
        assert "test-project" in message

    @patch("gforce.core.auth.google_auth_default")
    def test_unauthorized_message(self, mock_default):
        """Test message when not authenticated."""
        from google.auth.exceptions import DefaultCredentialsError

        mock_default.side_effect = DefaultCredentialsError()

        message = get_auth_status_message()

        assert "Not authenticated" in message


class TestAuthIntegration:
    """Integration-style tests for auth module."""

    def test_auth_flow_simulation(self):
        """Simulate a complete auth flow."""
        with patch("gforce.core.auth.google_auth_default") as mock_default:
            # Start with no auth
            from google.auth.exceptions import DefaultCredentialsError

            mock_default.side_effect = DefaultCredentialsError()
            assert check_auth_silent() is False

            # Auth is provided
            mock_creds = MagicMock()
            mock_creds.expired = False
            mock_creds.valid = True
            mock_default.side_effect = None
            mock_default.return_value = (mock_creds, "test-project")

            assert check_auth_silent() is True
            assert get_project_id() == "test-project"
