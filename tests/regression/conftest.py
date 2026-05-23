"""Shared fixtures for regression tests."""
import pytest
import pandas as pd
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def celebrity_df():
    return pd.read_csv(FIXTURES / "celebrity_people.csv")


@pytest.fixture
def ecommerce_df():
    return pd.read_csv(FIXTURES / "ecommerce_sales.csv")
