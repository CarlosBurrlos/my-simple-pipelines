"""Shared fixtures. Redirect the catalog's data paths into a tmp dir so tests
never write to the real data/ tree, and give a migrated ledger to work against."""

from __future__ import annotations

import pytest
from PIL import Image

from ledger import connect, migrate
from pipelines.catalog import config


@pytest.fixture
def catalog_env(tmp_path, monkeypatch):
    """Point every config data path at tmp_path/data and return that root."""
    data = tmp_path / "data"
    paths = {
        "DATA_DIR": data,
        "DB_PATH": data / "catalog.db",
        "LOGS_DIR": data / "logs",
        "INCOMING_DIR": data / "incoming",
        "ORIGINALS_DIR": data / "originals",
        "DERIVATIVES_DIR": data / "derivatives",
    }
    for attr, value in paths.items():
        monkeypatch.setattr(config, attr, value)
    config.ensure_data_dirs()
    return data


@pytest.fixture
def db(catalog_env):
    migrate(config.DB_PATH)
    conn = connect(config.DB_PATH)
    yield conn
    conn.close()


def make_image(path, size=(2200, 2200), color=(60, 140, 80)):
    """Write a solid-colour PNG (default large enough to pass intake)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)
    return path
