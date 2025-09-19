"""Frozen Parameter Registry for Reproducible Runs."""

from __future__ import annotations
import hashlib
import json
import pathlib
import subprocess
import sys
from dataclasses import dataclass, asdict
from typing import Dict, Any


def _hash_file(path: str) -> str:
    """Calculate SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> str:
    """Get current git commit hash."""
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
    except Exception:
        return 'unknown'


@dataclass(frozen=True)
class FrozenParams:
    """Immutable parameter record for reproducible runs."""
    strategy_name: str
    params: Dict[str, Any]
    random_seed: int
    requirements_sha256: str
    python_version: str
    git_commit: str


class ParameterRegistry:
    """Manages frozen parameter records for reproducible validation runs."""
    
    def __init__(self, out_dir: str = 'reports/params'):
        self.out = pathlib.Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)

    def freeze(self, strategy_name: str, params: Dict[str, Any], random_seed: int, 
               req_path: str = 'requirements.txt') -> str:
        """
        Freeze parameters for a strategy run.
        
        Args:
            strategy_name: Name of the strategy
            params: Strategy parameters
            random_seed: Random seed used
            req_path: Path to requirements file
            
        Returns:
            Path to the frozen parameter file
        """
        record = FrozenParams(
            strategy_name=strategy_name,
            params=params,
            random_seed=random_seed,
            requirements_sha256=_hash_file(req_path) if pathlib.Path(req_path).exists() else 'missing',
            python_version=sys.version,
            git_commit=_git_commit()
        )
        
        p = self.out / f'{strategy_name}_frozen.json'
        p.write_text(json.dumps(asdict(record), indent=2))
        return str(p)
    
    def load_frozen(self, strategy_name: str) -> FrozenParams | None:
        """Load frozen parameters for a strategy."""
        p = self.out / f'{strategy_name}_frozen.json'
        if not p.exists():
            return None
        
        data = json.loads(p.read_text())
        return FrozenParams(**data)
    
    def verify_reproducibility(self, strategy_name: str, current_params: Dict[str, Any], 
                              current_seed: int) -> Dict[str, Any]:
        """Verify current run matches frozen parameters."""
        frozen = self.load_frozen(strategy_name)
        if not frozen:
            return {'status': 'no_frozen_record', 'reproducible': False}
        
        params_match = frozen.params == current_params
        seed_match = frozen.random_seed == current_seed
        
        return {
            'status': 'verified' if params_match and seed_match else 'mismatch',
            'reproducible': params_match and seed_match,
            'params_match': params_match,
            'seed_match': seed_match,
            'frozen_record': frozen
        }


# Example usage and testing
if __name__ == "__main__":
    def test_parameter_registry():
        """Test the parameter registry."""
        print("=== Parameter Registry Test ===")
        
        # Create test requirements file
        test_req_path = '/tmp/test_requirements.txt'
        with open(test_req_path, 'w') as f:
            f.write('pandas==2.2.2\nnumpy==1.24.0\n')
        
        # Test freezing
        registry = ParameterRegistry(out_dir='/tmp/test_params')
        params = {'lookback': 20, 'threshold': 0.05}
        frozen_path = registry.freeze('TestStrategy', params, 42, test_req_path)
        
        print(f"Frozen parameters saved to: {frozen_path}")
        
        # Test loading
        loaded = registry.load_frozen('TestStrategy')
        if loaded:
            print(f"Loaded strategy: {loaded.strategy_name}")
            print(f"Parameters: {loaded.params}")
            print(f"Random seed: {loaded.random_seed}")
            print(f"Git commit: {loaded.git_commit}")
        
        # Test reproducibility verification
        verification = registry.verify_reproducibility('TestStrategy', params, 42)
        print(f"Reproducibility check: {verification}")
        
        # Cleanup
        import os
        os.remove(test_req_path)
        os.remove(frozen_path)
    
    test_parameter_registry()


