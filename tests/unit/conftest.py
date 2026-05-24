"""
Pytest configuration for unit tests.
Adds engine/ to sys.path so modules can be imported directly.
"""
import sys
import os

ENGINE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "engine")
sys.path.insert(0, ENGINE_DIR)
