from pathlib import Path
import sys

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from invoice_processor.config import config as config_mod  # noqa: E402
from invoice_processor.config.config import load_allowed_users, is_user_allowed  # noqa: E402


def write_yaml(path: Path, payload: dict):
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


class FakeSettings:
    def __init__(self, allowed_users_file: Path):
        self.allowed_users_file = allowed_users_file


def use_fake_settings(monkeypatch, file_path: Path):
    monkeypatch.setattr(config_mod, "get_settings", lambda: FakeSettings(file_path))


def test_load_allowed_users_valid(tmp_path, monkeypatch):
    cfg = {
        "slack_access": {
            "enable": True,
            "users": [
                {"id": "U123", "name": "alice"},
                {"id": "U456", "name": "bob"},
            ],
        }
    }
    file_path = tmp_path / "allowed.yaml"
    write_yaml(file_path, cfg)
    use_fake_settings(monkeypatch, file_path)
    info = load_allowed_users()

    assert info["enabled"] is True
    assert len(info["users"]) == 2
    assert info["users"][0]["id"] == "U123"


def test_load_allowed_users_invalid_entries(tmp_path, monkeypatch):
    cfg = {
        "slack_access": {
            "enabled": True,
            "users": [
                "bad",
                {"name": "missing id"},
                {"id": "U123"},
            ],
        }
    }
    file_path = tmp_path / "allowed.yaml"
    write_yaml(file_path, cfg)
    use_fake_settings(monkeypatch, file_path)
    info = load_allowed_users()

    # Solo 1 entrada válida (con id)
    assert len(info["users"]) == 1
    assert info["users"][0]["id"] == "U123"


def test_is_user_allowed(tmp_path, monkeypatch):
    cfg = {
        "slack_access": {
            "enabled": True,
            "users": [
                {"id": "U999", "name": "ok"},
            ],
        }
    }
    file_path = tmp_path / "allowed.yaml"
    write_yaml(file_path, cfg)
    use_fake_settings(monkeypatch, file_path)
    assert is_user_allowed("U999") is True
    assert is_user_allowed("U000") is False


def test_disabled_returns_false(tmp_path, monkeypatch):
    cfg = {
        "slack_access": {
            "enabled": False,
            "users": [
                {"id": "U111", "name": "x"},
            ],
        }
    }
    file_path = tmp_path / "allowed.yaml"
    write_yaml(file_path, cfg)
    use_fake_settings(monkeypatch, file_path)
    assert is_user_allowed("U111") is False
