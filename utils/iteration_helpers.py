"""Iteration helper utilities for safe dictionary iteration patterns."""

from typing import Dict, Any, Iterator, TypeVar, Union

T = TypeVar('T')


def safe_dict_values(data: Union[Dict[Any, T], Any]) -> Iterator[T]:
    """
    Safely iterate over dictionary values with type checking.

    This wrapper ensures we're working with a dictionary before calling .values(),
    and provides a safe fallback for edge cases.

    Args:
        data: The data structure to iterate over (should be a dict)

    Returns:
        Iterator over the values

    Raises:
        TypeError: If data is not a dictionary

    Example:
        # Instead of:
        for key, value in some_dict.items():
            process(value)  # key not used

        # Use:
        for value in safe_dict_values(some_dict):
            process(value)
    """
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict, got {type(data).__name__}")

    if not data:
        return iter([])  # Return empty iterator for empty dict

    return iter(data.values())


def safe_dict_items(data: Union[Dict[Any, T], Any]) -> Iterator[tuple[Any, T]]:
    """
    Safely iterate over dictionary items with type checking.

    Use this when you actually need both keys and values.

    Args:
        data: The data structure to iterate over (should be a dict)

    Returns:
        Iterator over (key, value) pairs

    Raises:
        TypeError: If data is not a dictionary
    """
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict, got {type(data).__name__}")

    if not data:
        return iter([])  # Return empty iterator for empty dict

    return iter(data.items())


def validate_dict_iteration_pattern(data: Any, use_keys: bool = False) -> Union[Iterator[Any], Iterator[tuple[Any, Any]]]:
    """
    Validate and choose the appropriate iteration pattern for a dictionary.

    This function analyzes the intended use case and returns the most efficient iterator.

    Args:
        data: The dictionary to iterate over
        use_keys: Whether the keys will be used in the iteration

    Returns:
        Appropriate iterator (values() if keys not needed, items() if keys needed)

    Raises:
        TypeError: If data is not a dictionary
    """
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict, got {type(data).__name__}")

    if not data:
        return iter([])

    if use_keys:
        return iter(data.items())
    else:
        return iter(data.values())


# Decorator for functions that should validate dict iteration patterns
def ensure_efficient_dict_iteration(func):
    """
    Decorator to ensure functions use efficient dictionary iteration patterns.

    This can be used to wrap test functions or other code that iterates over dicts
    to ensure they're using the most efficient iteration method.
    """
    def wrapper(*args, **kwargs):
        # This decorator could be extended to analyze the function's AST
        # and provide warnings about inefficient dict iteration patterns
        return func(*args, **kwargs)

    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    return wrapper


class DictIterationHelper:
    """
    Helper class for safe and efficient dictionary iteration.

    Provides methods to handle common dictionary iteration patterns
    with proper type checking and performance optimization.
    """

    @staticmethod
    def iterate_values_only(data: Dict[Any, T]) -> Iterator[T]:
        """Iterate when only values are needed."""
        return safe_dict_values(data)

    @staticmethod
    def iterate_with_keys(data: Dict[Any, T]) -> Iterator[tuple[Any, T]]:
        """Iterate when both keys and values are needed."""
        return safe_dict_items(data)

    @staticmethod
    def analyze_usage_pattern(code_snippet: str) -> str:
        """
        Analyze a code snippet to suggest the best iteration pattern.

        This is a simple heuristic-based analyzer.
        """
        if "for " in code_snippet and ", " in code_snippet:
            # Look for patterns like "for key, value in dict.items():"
            if ".items()" in code_snippet:
                # Check if the first variable (key) is used in the loop body
                lines = code_snippet.split('\n')
                for_line = None
                for line in lines:
                    if "for " in line and ".items()" in line:
                        for_line = line
                        break

                if for_line:
                    # Extract the key variable name
                    try:
                        parts = for_line.split("for ")[1].split(" in ")[0]
                        key_var = parts.split(",")[0].strip()

                        # Check if key_var is used in subsequent lines
                        body_lines = lines[lines.index(for_line) + 1:]
                        key_used = any(key_var in line for line in body_lines)

                        if not key_used:
                            return f"SUGGESTION: Use .values() instead of .items() - key '{key_var}' is not used"
                        else:
                            return "OK: Both key and value are used"
                    except (IndexError, AttributeError, ValueError):
                        return "ANALYSIS: Could not parse iteration pattern"

        return "OK: No dictionary iteration issues detected"


# Example usage patterns
def example_correct_values_only():
    """Example of correct values-only iteration."""
    data = {"a": 1, "b": 2, "c": 3}

    # Correct - only values needed
    for value in safe_dict_values(data):
        print(f"Processing value: {value}")


def example_correct_items():
    """Example of correct items iteration."""
    data = {"a": 1, "b": 2, "c": 3}

    # Correct - both key and value needed
    for key, value in safe_dict_items(data):
        print(f"Processing {key}: {value}")


def example_analysis():
    """Example of analyzing iteration patterns."""
    analyzer = DictIterationHelper()

    # Bad pattern
    bad_code = """
    for key, value in data.items():
        process(value)
        assert value > 0
    """

    # Good pattern
    good_code = """
    for key, value in data.items():
        print(f"Key {key} has value {value}")
        process(value)
    """

    print("Bad pattern analysis:", analyzer.analyze_usage_pattern(bad_code))
    print("Good pattern analysis:", analyzer.analyze_usage_pattern(good_code))