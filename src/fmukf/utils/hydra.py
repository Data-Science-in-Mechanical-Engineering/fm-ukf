"""
Custom OmegaConf resolvers for Hydra configuration.
This module provides reusable resolvers that can be imported and registered in any Hydra script.
"""

import os
import importlib.util
from omegaconf import OmegaConf


def floatop(x, op, y):
    """Float arithmetic operations for OmegaConf resolvers."""
    # Convert inputs to float
    a = float(x)
    b = float(y)
    if op == '+':
        return a + b
    elif op == '-':
        return a - b
    elif op == '*':
        return a * b
    elif op == '/':
        return a / b
    else:
        raise ValueError(f"Unsupported operator: {op}")


def intop(x, op, y):
    """Integer arithmetic operations for OmegaConf resolvers."""
    # Convert inputs to int
    a = int(x)
    b = int(y)
    if op == '+':
        return a + b
    elif op == '-':
        return a - b
    elif op == '*':
        return a * b
    elif op == '/':
        return a // b  # Integer division
    else:
        raise ValueError(f"Unsupported operator: {op}")


def pkgfile_resolver(package_name, *args):
    """Get the directory path of a Python package."""
    try:
        spec = importlib.util.find_spec(package_name)
        if spec and spec.origin:
            return os.path.dirname(spec.origin)
        else:
            raise ValueError(f"Package {package_name} not found")
    except Exception as e:
        raise ValueError(f"Could not resolve package path for {package_name}: {e}")


def register_fmukf_resolvers():
    """Register all custom fmukf resolvers with OmegaConf."""
    OmegaConf.register_new_resolver("floatop", floatop)
    OmegaConf.register_new_resolver("intop", intop)
    OmegaConf.register_new_resolver("pkgfile", pkgfile_resolver) 