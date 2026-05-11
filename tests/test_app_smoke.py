"""Smoke test: app.py imports cleanly and key bits work."""

import importlib


def test_app_imports():
    # Just importing should not raise
    importlib.import_module("app")


def test_attack_presets_have_required_keys():
    app = importlib.import_module("app")
    for label, preset in app._ATTACK_PRESETS.items():
        assert "position" in preset
        assert "velocity" in preset
        assert "window_fn" in preset
        assert "score_fn" in preset
