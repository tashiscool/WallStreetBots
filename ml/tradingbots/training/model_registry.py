"""
Model Registry — Versioned model storage with filesystem + JSON manifest.

Provides model lifecycle management: register → shadow → promote → retire.
"""

import json
import logging
import shutil
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ModelVersion:
    """A single versioned model snapshot."""

    version_id: str
    agent_type: str  # 'ppo', 'dqn', 'sac', etc.
    model_path: str  # path to saved checkpoint
    status: str = "candidate"  # candidate → shadow → active → retired
    created_at: str = ""
    training_metrics: Dict[str, float] = field(default_factory=dict)
    validation_metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.utcnow().isoformat()


class ModelRegistry:
    """
    Filesystem-backed registry for versioned RL model storage.

    Structure::

        base_dir/
            manifest.json          # {versions: [...], active_version_id: ...}
            models/
                v_001/checkpoint.pt
                v_002/checkpoint.pt
    """

    def __init__(self, base_dir: str) -> None:
        self.base_dir = Path(base_dir)
        self.models_dir = self.base_dir / "models"
        self.manifest_path = self.base_dir / "manifest.json"
        self._ensure_dirs()
        self._manifest = self._load_manifest()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(
        self,
        agent_type: str,
        model_path: str,
        training_metrics: Optional[Dict[str, float]] = None,
        validation_metrics: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ModelVersion:
        """Register a new model version as *candidate*."""
        version_id = self._next_version_id()
        dest_dir = self.models_dir / version_id
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Copy checkpoint into registry
        src = Path(model_path)
        dest = dest_dir / src.name
        shutil.copy2(str(src), str(dest))

        version = ModelVersion(
            version_id=version_id,
            agent_type=agent_type,
            model_path=str(dest),
            status="candidate",
            training_metrics=training_metrics or {},
            validation_metrics=validation_metrics or {},
            metadata=metadata or {},
        )
        self._manifest["versions"].append(asdict(version))
        self._save_manifest()
        logger.info("Registered model %s (agent_type=%s)", version_id, agent_type)
        return version

    def get_active(self) -> Optional[ModelVersion]:
        """Return the currently active model version, or None."""
        active_id = self._manifest.get("active_version_id")
        if active_id is None:
            return None
        return self._find_version(active_id)

    def get_version(self, version_id: str) -> Optional[ModelVersion]:
        """Get a specific version by ID."""
        return self._find_version(version_id)

    def list_versions(self, status: Optional[str] = None) -> List[ModelVersion]:
        """List versions, optionally filtered by status."""
        versions = [
            ModelVersion(**v) for v in self._manifest["versions"]
        ]
        if status:
            versions = [v for v in versions if v.status == status]
        return versions

    def promote(self, version_id: str) -> ModelVersion:
        """Promote a candidate/shadow version to *active*, retiring the old active."""
        version = self._find_version(version_id)
        if version is None:
            raise ValueError(f"Version {version_id} not found")

        # Retire current active
        old_active = self.get_active()
        if old_active is not None:
            self._set_status(old_active.version_id, "retired")

        self._set_status(version_id, "active")
        self._manifest["active_version_id"] = version_id
        self._save_manifest()
        logger.info("Promoted %s to active (retired %s)",
                     version_id, old_active.version_id if old_active else "none")
        return self._find_version(version_id)

    def rollback(self) -> Optional[ModelVersion]:
        """Roll back to the most recent retired version."""
        retired = [
            v for v in self.list_versions()
            if v.status == "retired"
        ]
        if not retired:
            logger.warning("No retired versions available for rollback")
            return None

        # Sort by created_at descending, pick most recent
        retired.sort(key=lambda v: v.created_at, reverse=True)
        target = retired[0]

        # Retire current active
        old_active = self.get_active()
        if old_active is not None:
            self._set_status(old_active.version_id, "retired")

        self._set_status(target.version_id, "active")
        self._manifest["active_version_id"] = target.version_id
        self._save_manifest()
        logger.info("Rolled back to %s", target.version_id)
        return self._find_version(target.version_id)

    def load_agent(
        self,
        version_id: str,
        state_dim: int,
        action_dim: int,
        **kwargs,
    ) -> Any:
        """Instantiate an RL agent from a stored checkpoint.

        Uses ``create_rl_agent`` factory + ``agent.load()``.
        """
        from ..components.rl_agents import create_rl_agent

        version = self._find_version(version_id)
        if version is None:
            raise ValueError(f"Version {version_id} not found")

        agent = create_rl_agent(
            version.agent_type, state_dim, action_dim, **kwargs
        )
        agent.load(version.model_path)
        return agent

    def cleanup(self, keep_last_n: int = 5) -> int:
        """Remove retired versions older than the most recent *keep_last_n*."""
        retired = [v for v in self.list_versions() if v.status == "retired"]
        retired.sort(key=lambda v: v.created_at, reverse=True)

        to_remove = retired[keep_last_n:]
        removed = 0
        for v in to_remove:
            version_dir = self.models_dir / v.version_id
            if version_dir.exists():
                shutil.rmtree(str(version_dir))
            self._manifest["versions"] = [
                mv for mv in self._manifest["versions"]
                if mv["version_id"] != v.version_id
            ]
            removed += 1

        if removed:
            self._save_manifest()
            logger.info("Cleaned up %d old versions", removed)
        return removed

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _ensure_dirs(self) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def _load_manifest(self) -> Dict[str, Any]:
        if self.manifest_path.exists():
            with open(self.manifest_path) as f:
                return json.load(f)
        return {"versions": [], "active_version_id": None}

    def _save_manifest(self) -> None:
        with open(self.manifest_path, "w") as f:
            json.dump(self._manifest, f, indent=2)

    def _next_version_id(self) -> str:
        existing = self._manifest["versions"]
        idx = len(existing) + 1
        return f"v_{idx:04d}"

    def _find_version(self, version_id: str) -> Optional[ModelVersion]:
        for v in self._manifest["versions"]:
            if v["version_id"] == version_id:
                return ModelVersion(**v)
        return None

    def _set_status(self, version_id: str, status: str) -> None:
        for v in self._manifest["versions"]:
            if v["version_id"] == version_id:
                v["status"] = status
                return
