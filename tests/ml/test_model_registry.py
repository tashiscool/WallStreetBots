"""Tests for ModelRegistry — versioned model storage."""

import json
import os
import tempfile

import pytest

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ml.tradingbots.training.model_registry import ModelRegistry, ModelVersion


@pytest.fixture
def registry_dir(tmp_path):
    return str(tmp_path / "registry")


@pytest.fixture
def registry(registry_dir):
    return ModelRegistry(registry_dir)


@pytest.fixture
def dummy_checkpoint(tmp_path):
    """Create a dummy checkpoint file."""
    path = tmp_path / "dummy_checkpoint.pt"
    path.write_text("fake-checkpoint-data")
    return str(path)


class TestModelVersion:
    def test_defaults(self):
        v = ModelVersion(version_id="v_0001", agent_type="ppo", model_path="/tmp/x")
        assert v.status == "candidate"
        assert v.created_at  # auto-set
        assert v.training_metrics == {}

    def test_custom_fields(self):
        v = ModelVersion(
            version_id="v_0002",
            agent_type="dqn",
            model_path="/tmp/y",
            status="active",
            training_metrics={"sharpe_ratio": 1.5},
        )
        assert v.status == "active"
        assert v.training_metrics["sharpe_ratio"] == 1.5


class TestRegister:
    def test_register_creates_version(self, registry, dummy_checkpoint):
        v = registry.register("ppo", dummy_checkpoint)
        assert v.version_id == "v_0001"
        assert v.agent_type == "ppo"
        assert v.status == "candidate"
        assert os.path.exists(v.model_path)

    def test_register_copies_checkpoint(self, registry, dummy_checkpoint):
        v = registry.register("ppo", dummy_checkpoint)
        with open(v.model_path) as f:
            assert f.read() == "fake-checkpoint-data"

    def test_register_with_metrics(self, registry, dummy_checkpoint):
        v = registry.register(
            "dqn",
            dummy_checkpoint,
            training_metrics={"sharpe_ratio": 1.2},
            validation_metrics={"max_drawdown": 0.05},
        )
        assert v.training_metrics["sharpe_ratio"] == 1.2
        assert v.validation_metrics["max_drawdown"] == 0.05

    def test_register_multiple(self, registry, dummy_checkpoint):
        v1 = registry.register("ppo", dummy_checkpoint)
        v2 = registry.register("dqn", dummy_checkpoint)
        assert v1.version_id == "v_0001"
        assert v2.version_id == "v_0002"


class TestPromote:
    def test_promote_candidate_to_active(self, registry, dummy_checkpoint):
        v = registry.register("ppo", dummy_checkpoint)
        promoted = registry.promote(v.version_id)
        assert promoted.status == "active"
        assert registry.get_active().version_id == v.version_id

    def test_promote_retires_old_active(self, registry, dummy_checkpoint):
        v1 = registry.register("ppo", dummy_checkpoint)
        registry.promote(v1.version_id)

        v2 = registry.register("ppo", dummy_checkpoint)
        registry.promote(v2.version_id)

        active = registry.get_active()
        assert active.version_id == v2.version_id

        old = registry.get_version(v1.version_id)
        assert old.status == "retired"

    def test_promote_nonexistent_raises(self, registry):
        with pytest.raises(ValueError, match="not found"):
            registry.promote("v_9999")


class TestRollback:
    def test_rollback_to_previous(self, registry, dummy_checkpoint):
        v1 = registry.register("ppo", dummy_checkpoint)
        registry.promote(v1.version_id)

        v2 = registry.register("ppo", dummy_checkpoint)
        registry.promote(v2.version_id)

        rolled = registry.rollback()
        assert rolled.version_id == v1.version_id
        assert rolled.status == "active"

    def test_rollback_no_retired(self, registry, dummy_checkpoint):
        v1 = registry.register("ppo", dummy_checkpoint)
        registry.promote(v1.version_id)
        result = registry.rollback()
        assert result is None


class TestGetActive:
    def test_no_active_initially(self, registry):
        assert registry.get_active() is None

    def test_returns_active(self, registry, dummy_checkpoint):
        v = registry.register("ppo", dummy_checkpoint)
        registry.promote(v.version_id)
        assert registry.get_active().version_id == v.version_id


class TestListVersions:
    def test_list_all(self, registry, dummy_checkpoint):
        registry.register("ppo", dummy_checkpoint)
        registry.register("dqn", dummy_checkpoint)
        assert len(registry.list_versions()) == 2

    def test_filter_by_status(self, registry, dummy_checkpoint):
        v1 = registry.register("ppo", dummy_checkpoint)
        registry.register("dqn", dummy_checkpoint)
        registry.promote(v1.version_id)
        assert len(registry.list_versions(status="active")) == 1
        assert len(registry.list_versions(status="candidate")) == 1


class TestCleanup:
    def test_cleanup_keeps_recent(self, registry, dummy_checkpoint):
        for _ in range(8):
            v = registry.register("ppo", dummy_checkpoint)
            registry.promote(v.version_id)

        # All but last should be retired
        removed = registry.cleanup(keep_last_n=3)
        # 7 retired, keep 3 → remove 4
        assert removed == 4
        assert len(registry.list_versions()) == 4  # 3 retired + 1 active

    def test_cleanup_nothing_to_remove(self, registry, dummy_checkpoint):
        v = registry.register("ppo", dummy_checkpoint)
        registry.promote(v.version_id)
        assert registry.cleanup(keep_last_n=5) == 0


class TestPersistence:
    def test_manifest_persists(self, registry_dir, dummy_checkpoint):
        reg1 = ModelRegistry(registry_dir)
        v = reg1.register("ppo", dummy_checkpoint)
        reg1.promote(v.version_id)

        # New registry instance reads from disk
        reg2 = ModelRegistry(registry_dir)
        active = reg2.get_active()
        assert active is not None
        assert active.version_id == v.version_id
