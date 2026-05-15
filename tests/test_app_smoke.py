"""Smoke test: app.py imports cleanly and key bits work."""

import importlib


def test_app_imports():
    # Just importing should not raise
    importlib.import_module("app")


def test_attack_presets_keyed_by_position_and_type():
    app = importlib.import_module("app")
    for key, preset in app._ATTACK_PRESETS.items():
        position, attack_name = key
        assert position in {"4号位", "2号位"}
        assert attack_name in app._SCORE_FN
        assert "position" in preset
        assert "velocity" in preset


def test_score_and_window_factories_cover_all_keys():
    app = importlib.import_module("app")
    for position, _ in app._ATTACK_PRESETS:
        assert position in app._WINDOW_FACTORY
    # Every attack name in score-fn map appears in at least one preset
    used_attack_names = {a for _, a in app._ATTACK_PRESETS}
    assert used_attack_names == set(app._SCORE_FN)
