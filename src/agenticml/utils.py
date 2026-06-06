"""
utility functions for the agenticml library.
"""

def format_reward(x: float) -> str:
    """format a reward number compactly without trailing zeros."""
    if isinstance(x, int) or float(x).is_integer():
        return str(int(x))
    return repr(float(x))