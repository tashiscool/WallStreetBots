"""Tests for Parameter Registry."""

import pytest
import tempfile
import json
import os
import hashlib
import subprocess
from pathlib import Path
from unittest.mock import patch, mock_open, Mock
from backend.validation.parameter_registry import ParameterRegistry, FrozenParams, _hash_file, _git_commit


def test_freeze(tmp_path, monkeypatch):
    """Test parameter freezing."""
    # Create test requirements file
    req_file = tmp_path / 'requirements.txt'
    req_file.write_text('pandas==2.2.2\nnumpy==1.24.0\n')
    
    # Test freezing
    pr = ParameterRegistry(out_dir=str(tmp_path / 'out'))
    path = pr.freeze('IndexBaseline', {'lookback':20}, 42, req_path=str(req_file))
    
    assert 'IndexBaseline_frozen.json' in path
    assert Path(path).exists()
    
    # Verify content
    data = json.loads(Path(path).read_text())
    assert data['strategy_name'] == 'IndexBaseline'
    assert data['params'] == {'lookback': 20}
    assert data['random_seed'] == 42
    assert data['requirements_sha256'] != 'missing'


def test_load_frozen(tmp_path):
    """Test loading frozen parameters."""
    # Create frozen file
    frozen_data = {
        'strategy_name': 'TestStrategy',
        'params': {'param1': 'value1'},
        'random_seed': 123,
        'requirements_sha256': 'abc123',
        'python_version': '3.12.0',
        'git_commit': 'def456'
    }
    
    frozen_file = tmp_path / 'TestStrategy_frozen.json'
    frozen_file.write_text(json.dumps(frozen_data))
    
    # Test loading
    pr = ParameterRegistry(out_dir=str(tmp_path))
    loaded = pr.load_frozen('TestStrategy')
    
    assert loaded is not None
    assert loaded.strategy_name == 'TestStrategy'
    assert loaded.params == {'param1': 'value1'}
    assert loaded.random_seed == 123


def test_load_nonexistent():
    """Test loading non-existent frozen parameters."""
    pr = ParameterRegistry()
    loaded = pr.load_frozen('NonExistentStrategy')
    assert loaded is None


def test_verify_reproducibility(tmp_path):
    """Test reproducibility verification."""
    # Create frozen file
    frozen_data = {
        'strategy_name': 'TestStrategy',
        'params': {'param1': 'value1'},
        'random_seed': 123,
        'requirements_sha256': 'abc123',
        'python_version': '3.12.0',
        'git_commit': 'def456'
    }
    
    frozen_file = tmp_path / 'TestStrategy_frozen.json'
    frozen_file.write_text(json.dumps(frozen_data))
    
    pr = ParameterRegistry(out_dir=str(tmp_path))
    
    # Test matching parameters
    verification = pr.verify_reproducibility('TestStrategy', {'param1': 'value1'}, 123)
    assert verification['reproducible'] is True
    assert verification['params_match'] is True
    assert verification['seed_match'] is True
    
    # Test mismatched parameters
    verification = pr.verify_reproducibility('TestStrategy', {'param1': 'different'}, 123)
    assert verification['reproducible'] is False
    assert verification['params_match'] is False
    assert verification['seed_match'] is True
    
    # Test mismatched seed
    verification = pr.verify_reproducibility('TestStrategy', {'param1': 'value1'}, 456)
    assert verification['reproducible'] is False
    assert verification['params_match'] is True
    assert verification['seed_match'] is False


def test_verify_nonexistent():
    """Test verification with non-existent strategy."""
    pr = ParameterRegistry()
    verification = pr.verify_reproducibility('NonExistent', {}, 0)
    assert verification['status'] == 'no_frozen_record'
    assert verification['reproducible'] is False


def test_frozen_params_immutability():
    """Test that FrozenParams is truly immutable."""
    params = FrozenParams(
        strategy_name='Test',
        params={'key': 'value'},
        random_seed=42,
        requirements_sha256='abc123',
        python_version='3.12.0',
        git_commit='def456'
    )
    
    # Should not be able to modify
    with pytest.raises(AttributeError):
        params.strategy_name = 'Modified'
    
    # The dataclass itself is frozen, but the dict contents can still be modified
    # This is expected behavior - the dict reference is frozen, not the contents
    params.params['key'] = 'modified'  # This should work
    assert params.params['key'] == 'modified'


class TestHashFile:
    """Test _hash_file utility function."""

    def test_hash_file_basic(self, tmp_path):
        """Test basic file hashing."""
        test_file = tmp_path / "test.txt"
        test_content = "Hello, world!"
        test_file.write_text(test_content)

        result = _hash_file(str(test_file))

        # Calculate expected hash
        expected = hashlib.sha256(test_content.encode()).hexdigest()
        assert result == expected

    def test_hash_file_binary(self, tmp_path):
        """Test hashing of binary files."""
        test_file = tmp_path / "test.bin"
        test_content = b"\x00\x01\x02\x03\xff\xfe\xfd"
        test_file.write_bytes(test_content)

        result = _hash_file(str(test_file))

        # Calculate expected hash
        expected = hashlib.sha256(test_content).hexdigest()
        assert result == expected

    def test_hash_file_large(self, tmp_path):
        """Test hashing of large files (chunked reading)."""
        test_file = tmp_path / "large.txt"
        # Create file larger than chunk size (65536)
        large_content = "A" * 100000
        test_file.write_text(large_content)

        result = _hash_file(str(test_file))

        # Calculate expected hash
        expected = hashlib.sha256(large_content.encode()).hexdigest()
        assert result == expected

    def test_hash_file_empty(self, tmp_path):
        """Test hashing of empty files."""
        test_file = tmp_path / "empty.txt"
        test_file.write_text("")

        result = _hash_file(str(test_file))

        # Hash of empty content
        expected = hashlib.sha256(b"").hexdigest()
        assert result == expected

    def test_hash_file_nonexistent(self):
        """Test hashing of non-existent file."""
        with pytest.raises(FileNotFoundError):
            _hash_file("/nonexistent/file.txt")

    def test_hash_file_permission_denied(self, tmp_path):
        """Test hashing with permission denied."""
        test_file = tmp_path / "restricted.txt"
        test_file.write_text("content")

        # Make file unreadable
        test_file.chmod(0o000)

        try:
            with pytest.raises(PermissionError):
                _hash_file(str(test_file))
        finally:
            # Restore permissions for cleanup
            test_file.chmod(0o644)

    def test_hash_file_consistency(self, tmp_path):
        """Test that hash is consistent across multiple calls."""
        test_file = tmp_path / "consistent.txt"
        test_file.write_text("consistent content")

        hash1 = _hash_file(str(test_file))
        hash2 = _hash_file(str(test_file))

        assert hash1 == hash2

    def test_hash_file_different_content(self, tmp_path):
        """Test that different content produces different hashes."""
        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"

        file1.write_text("content1")
        file2.write_text("content2")

        hash1 = _hash_file(str(file1))
        hash2 = _hash_file(str(file2))

        assert hash1 != hash2


class TestGitCommit:
    """Test _git_commit utility function."""

    @patch('subprocess.check_output')
    def test_git_commit_success(self, mock_subprocess):
        """Test successful git commit retrieval."""
        mock_subprocess.return_value = b"abc123def456\n"

        result = _git_commit()

        assert result == "abc123def456"
        mock_subprocess.assert_called_once_with(['git', 'rev-parse', 'HEAD'])

    @patch('subprocess.check_output')
    def test_git_commit_subprocess_error(self, mock_subprocess):
        """Test git command failure."""
        mock_subprocess.side_effect = subprocess.CalledProcessError(1, 'git')

        result = _git_commit()

        assert result == 'unknown'

    @patch('subprocess.check_output')
    def test_git_commit_file_not_found(self, mock_subprocess):
        """Test git command not found."""
        mock_subprocess.side_effect = FileNotFoundError()

        result = _git_commit()

        assert result == 'unknown'

    @patch('subprocess.check_output')
    def test_git_commit_permission_error(self, mock_subprocess):
        """Test git command permission error."""
        mock_subprocess.side_effect = PermissionError()

        result = _git_commit()

        assert result == 'unknown'

    @patch('subprocess.check_output')
    def test_git_commit_unicode_decode_error(self, mock_subprocess):
        """Test git command with invalid unicode output."""
        mock_subprocess.side_effect = UnicodeDecodeError('utf-8', b'\xff\xfe', 0, 2, 'invalid start byte')

        result = _git_commit()

        assert result == 'unknown'

    @patch('subprocess.check_output')
    def test_git_commit_with_whitespace(self, mock_subprocess):
        """Test git commit with leading/trailing whitespace."""
        mock_subprocess.return_value = b"  abc123def456  \n\r\t  "

        result = _git_commit()

        assert result == "abc123def456"


class TestParameterRegistryEdgeCases:
    """Test edge cases for ParameterRegistry."""

    def test_registry_creation_with_invalid_path(self):
        """Test registry creation with invalid path."""
        # This should raise an error when the path cannot be created
        with pytest.raises((OSError, PermissionError, FileNotFoundError)):
            ParameterRegistry(out_dir="/nonexistent/very/deep/path")

    def test_registry_creation_permission_denied(self, tmp_path):
        """Test registry creation with permission denied."""
        restricted_dir = tmp_path / "restricted"
        restricted_dir.mkdir()
        restricted_dir.chmod(0o000)  # No permissions

        try:
            # This might succeed or fail depending on the system
            # We test that it doesn't crash the application
            registry = ParameterRegistry(out_dir=str(restricted_dir / "subdir"))
            # If it succeeds, that's fine
            assert isinstance(registry, ParameterRegistry)
        except PermissionError:
            # If it fails with permission error, that's also expected
            pass
        finally:
            # Restore permissions for cleanup
            restricted_dir.chmod(0o755)

    def test_freeze_with_missing_requirements(self, tmp_path):
        """Test freezing with missing requirements file."""
        registry = ParameterRegistry(out_dir=str(tmp_path))
        params = {'test': 'value'}

        # Should not raise error, should set requirements_sha256 to 'missing'
        frozen_path = registry.freeze('TestStrategy', params, 42, req_path='/nonexistent/requirements.txt')

        # Verify the frozen file
        data = json.loads(Path(frozen_path).read_text())
        assert data['requirements_sha256'] == 'missing'

    def test_freeze_with_complex_parameters(self, tmp_path):
        """Test freezing with complex parameter types."""
        registry = ParameterRegistry(out_dir=str(tmp_path))

        complex_params = {
            'nested_dict': {'inner': {'deep': 'value'}},
            'list_param': [1, 2, 3, 'string', {'nested': True}],
            'none_value': None,
            'boolean': True,
            'float': 3.14159,
            'negative_int': -42
        }

        frozen_path = registry.freeze('ComplexStrategy', complex_params, 42)

        # Verify the frozen file can be loaded
        data = json.loads(Path(frozen_path).read_text())
        assert data['params'] == complex_params

    def test_freeze_with_large_parameters(self, tmp_path):
        """Test freezing with very large parameters."""
        registry = ParameterRegistry(out_dir=str(tmp_path))

        # Create large parameter dict (1MB of data)
        large_params = {
            'large_list': ['x' * 1000] * 1000,
            'large_dict': {f'key_{i}': f'value_{i}' * 100 for i in range(1000)}
        }

        frozen_path = registry.freeze('LargeStrategy', large_params, 42)

        # Verify the frozen file can be loaded
        loaded = registry.load_frozen('LargeStrategy')
        assert loaded.params == large_params

    def test_freeze_file_write_error(self, tmp_path):
        """Test freeze with file write error."""
        registry = ParameterRegistry(out_dir=str(tmp_path))

        # Make directory read-only
        tmp_path.chmod(0o444)

        try:
            with pytest.raises(PermissionError):
                registry.freeze('TestStrategy', {'test': 'value'}, 42)
        finally:
            # Restore permissions
            tmp_path.chmod(0o755)

    @patch('json.dumps')
    def test_freeze_json_serialization_error(self, mock_dumps, tmp_path):
        """Test freeze with JSON serialization error."""
        registry = ParameterRegistry(out_dir=str(tmp_path))
        mock_dumps.side_effect = TypeError("Object not JSON serializable")

        with pytest.raises(TypeError):
            registry.freeze('TestStrategy', {'test': 'value'}, 42)

    def test_load_frozen_with_corrupted_json(self, tmp_path):
        """Test loading frozen parameters with corrupted JSON."""
        registry = ParameterRegistry(out_dir=str(tmp_path))

        # Create corrupted JSON file
        corrupted_file = tmp_path / "CorruptedStrategy_frozen.json"
        corrupted_file.write_text("{invalid json content")

        with pytest.raises(json.JSONDecodeError):
            registry.load_frozen('CorruptedStrategy')

    def test_load_frozen_with_missing_fields(self, tmp_path):
        """Test loading frozen parameters with missing required fields."""
        registry = ParameterRegistry(out_dir=str(tmp_path))

        # Create JSON with missing fields
        incomplete_data = {
            'strategy_name': 'IncompleteStrategy',
            'params': {'test': 'value'}
            # Missing: random_seed, requirements_sha256, python_version, git_commit
        }

        incomplete_file = tmp_path / "IncompleteStrategy_frozen.json"
        incomplete_file.write_text(json.dumps(incomplete_data))

        with pytest.raises(TypeError):  # Missing required positional arguments
            registry.load_frozen('IncompleteStrategy')

    def test_load_frozen_with_extra_fields(self, tmp_path):
        """Test loading frozen parameters with extra fields."""
        registry = ParameterRegistry(out_dir=str(tmp_path))

        # Create JSON with extra fields
        extra_data = {
            'strategy_name': 'ExtraStrategy',
            'params': {'test': 'value'},
            'random_seed': 42,
            'requirements_sha256': 'abc123',
            'python_version': '3.12.0',
            'git_commit': 'def456',
            'extra_field': 'should_be_ignored'
        }

        extra_file = tmp_path / "ExtraStrategy_frozen.json"
        extra_file.write_text(json.dumps(extra_data))

        with pytest.raises(TypeError):  # Unexpected keyword argument
            registry.load_frozen('ExtraStrategy')

    def test_load_frozen_file_permission_error(self, tmp_path):
        """Test loading frozen parameters with file permission error."""
        registry = ParameterRegistry(out_dir=str(tmp_path))

        # Create file and make it unreadable
        frozen_file = tmp_path / "PermissionStrategy_frozen.json"
        frozen_file.write_text('{"test": "data"}')
        frozen_file.chmod(0o000)

        try:
            with pytest.raises(PermissionError):
                registry.load_frozen('PermissionStrategy')
        finally:
            # Restore permissions for cleanup
            frozen_file.chmod(0o644)

    def test_verify_reproducibility_with_none_types(self, tmp_path):
        """Test reproducibility verification with None values."""
        registry = ParameterRegistry(out_dir=str(tmp_path))

        # Create frozen file with None values
        frozen_data = {
            'strategy_name': 'NoneStrategy',
            'params': {'param1': None, 'param2': 'value'},
            'random_seed': 42,
            'requirements_sha256': 'abc123',
            'python_version': '3.12.0',
            'git_commit': 'def456'
        }

        frozen_file = tmp_path / "NoneStrategy_frozen.json"
        frozen_file.write_text(json.dumps(frozen_data))

        # Test with matching None values
        verification = registry.verify_reproducibility('NoneStrategy', {'param1': None, 'param2': 'value'}, 42)
        assert verification['reproducible'] is True

        # Test with mismatched None values
        verification = registry.verify_reproducibility('NoneStrategy', {'param1': 'not_none', 'param2': 'value'}, 42)
        assert verification['reproducible'] is False

    def test_verify_reproducibility_with_nested_structures(self, tmp_path):
        """Test reproducibility verification with nested data structures."""
        registry = ParameterRegistry(out_dir=str(tmp_path))

        complex_params = {
            'nested': {'level1': {'level2': [1, 2, {'level3': 'deep'}]}},
            'list': [{'item': 1}, {'item': 2}]
        }

        frozen_data = {
            'strategy_name': 'NestedStrategy',
            'params': complex_params,
            'random_seed': 42,
            'requirements_sha256': 'abc123',
            'python_version': '3.12.0',
            'git_commit': 'def456'
        }

        frozen_file = tmp_path / "NestedStrategy_frozen.json"
        frozen_file.write_text(json.dumps(frozen_data))

        # Test exact match
        verification = registry.verify_reproducibility('NestedStrategy', complex_params, 42)
        assert verification['reproducible'] is True

        # Test with different nested value
        modified_params = {
            'nested': {'level1': {'level2': [1, 2, {'level3': 'different'}]}},
            'list': [{'item': 1}, {'item': 2}]
        }
        verification = registry.verify_reproducibility('NestedStrategy', modified_params, 42)
        assert verification['reproducible'] is False

    def test_verify_reproducibility_return_structure(self, tmp_path):
        """Test that verify_reproducibility returns complete structure."""
        registry = ParameterRegistry(out_dir=str(tmp_path))

        frozen_data = {
            'strategy_name': 'StructureTest',
            'params': {'test': 'value'},
            'random_seed': 42,
            'requirements_sha256': 'abc123',
            'python_version': '3.12.0',
            'git_commit': 'def456'
        }

        frozen_file = tmp_path / "StructureTest_frozen.json"
        frozen_file.write_text(json.dumps(frozen_data))

        verification = registry.verify_reproducibility('StructureTest', {'test': 'value'}, 42)

        # Check all required keys are present
        required_keys = ['status', 'reproducible', 'params_match', 'seed_match', 'frozen_record']
        for key in required_keys:
            assert key in verification

        # Check types
        assert isinstance(verification['status'], str)
        assert isinstance(verification['reproducible'], bool)
        assert isinstance(verification['params_match'], bool)
        assert isinstance(verification['seed_match'], bool)
        assert isinstance(verification['frozen_record'], FrozenParams)


class TestFrozenParamsEdgeCases:
    """Test edge cases for FrozenParams dataclass."""

    def test_frozen_params_creation_with_all_types(self):
        """Test FrozenParams creation with various data types."""
        params = FrozenParams(
            strategy_name='ComplexTest',
            params={
                'string': 'value',
                'int': 42,
                'float': 3.14,
                'bool': True,
                'none': None,
                'list': [1, 2, 3],
                'dict': {'nested': 'value'}
            },
            random_seed=-123,  # Negative seed
            requirements_sha256='a' * 64,  # Full SHA256 length
            python_version='3.12.0 (main, Oct  2 2023, 12:00:00)',  # Full version string
            git_commit='1234567890abcdef' * 2  # Long commit hash
        )

        assert params.strategy_name == 'ComplexTest'
        assert params.random_seed == -123
        assert len(params.requirements_sha256) == 64

    def test_frozen_params_equality(self):
        """Test FrozenParams equality comparison."""
        params1 = FrozenParams(
            strategy_name='Test',
            params={'key': 'value'},
            random_seed=42,
            requirements_sha256='abc123',
            python_version='3.12.0',
            git_commit='def456'
        )

        params2 = FrozenParams(
            strategy_name='Test',
            params={'key': 'value'},
            random_seed=42,
            requirements_sha256='abc123',
            python_version='3.12.0',
            git_commit='def456'
        )

        params3 = FrozenParams(
            strategy_name='Test',
            params={'key': 'different'},  # Different params
            random_seed=42,
            requirements_sha256='abc123',
            python_version='3.12.0',
            git_commit='def456'
        )

        assert params1 == params2
        assert params1 != params3

    def test_frozen_params_repr(self):
        """Test FrozenParams string representation."""
        params = FrozenParams(
            strategy_name='TestRepr',
            params={'test': 'value'},
            random_seed=42,
            requirements_sha256='abc123',
            python_version='3.12.0',
            git_commit='def456'
        )

        repr_str = repr(params)
        assert 'FrozenParams' in repr_str
        assert 'TestRepr' in repr_str

    def test_frozen_params_hash(self):
        """Test that FrozenParams cannot be hashed due to containing dict."""
        params1 = FrozenParams(
            strategy_name='Test',
            params={'key': 'value'},
            random_seed=42,
            requirements_sha256='abc123',
            python_version='3.12.0',
            git_commit='def456'
        )

        # FrozenParams containing dict cannot be hashed
        with pytest.raises(TypeError, match="unhashable type"):
            hash(params1)

        # Cannot be used as dict key
        with pytest.raises(TypeError, match="unhashable type"):
            test_dict = {params1: 'value'}


class TestParameterRegistryIntegration:
    """Integration tests for complete workflows."""

    def test_complete_workflow_success(self, tmp_path):
        """Test complete freeze -> load -> verify workflow."""
        # Create requirements file
        req_file = tmp_path / 'requirements.txt'
        req_file.write_text('pandas==2.2.2\nnumpy==1.24.0\n')

        registry = ParameterRegistry(out_dir=str(tmp_path))
        params = {
            'strategy': 'momentum',
            'lookback': 20,
            'threshold': 0.05,
            'instruments': ['SPY', 'QQQ', 'IWM']
        }

        # Freeze parameters
        frozen_path = registry.freeze('MomentumStrategy', params, 12345, req_path=str(req_file))
        assert Path(frozen_path).exists()

        # Load parameters
        loaded = registry.load_frozen('MomentumStrategy')
        assert loaded is not None
        assert loaded.strategy_name == 'MomentumStrategy'
        assert loaded.params == params
        assert loaded.random_seed == 12345

        # Verify reproducibility
        verification = registry.verify_reproducibility('MomentumStrategy', params, 12345)
        assert verification['reproducible'] is True
        assert verification['status'] == 'verified'

    def test_workflow_with_parameter_drift(self, tmp_path):
        """Test workflow when parameters have drifted."""
        registry = ParameterRegistry(out_dir=str(tmp_path))
        original_params = {'lookback': 20, 'threshold': 0.05}

        # Freeze original parameters
        registry.freeze('DriftTest', original_params, 42)

        # Try to verify with different parameters
        drifted_params = {'lookback': 30, 'threshold': 0.05}  # Changed lookback
        verification = registry.verify_reproducibility('DriftTest', drifted_params, 42)

        assert verification['reproducible'] is False
        assert verification['status'] == 'mismatch'
        assert verification['params_match'] is False
        assert verification['seed_match'] is True

    def test_workflow_with_seed_drift(self, tmp_path):
        """Test workflow when random seed has drifted."""
        registry = ParameterRegistry(out_dir=str(tmp_path))
        params = {'lookback': 20, 'threshold': 0.05}

        # Freeze with original seed
        registry.freeze('SeedDriftTest', params, 42)

        # Try to verify with different seed
        verification = registry.verify_reproducibility('SeedDriftTest', params, 123)  # Different seed

        assert verification['reproducible'] is False
        assert verification['status'] == 'mismatch'
        assert verification['params_match'] is True
        assert verification['seed_match'] is False

    def test_multiple_strategies_same_registry(self, tmp_path):
        """Test multiple strategies in the same registry."""
        registry = ParameterRegistry(out_dir=str(tmp_path))

        # Freeze multiple strategies
        strategies = [
            ('Momentum', {'lookback': 20}, 42),
            ('MeanReversion', {'window': 50}, 123),
            ('Arbitrage', {'spread_threshold': 0.01}, 456)
        ]

        for name, params, seed in strategies:
            registry.freeze(name, params, seed)

        # Verify all can be loaded
        for name, expected_params, expected_seed in strategies:
            loaded = registry.load_frozen(name)
            assert loaded is not None
            assert loaded.strategy_name == name
            assert loaded.params == expected_params
            assert loaded.random_seed == expected_seed

            verification = registry.verify_reproducibility(name, expected_params, expected_seed)
            assert verification['reproducible'] is True
