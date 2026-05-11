import sys
sys.path.insert(0, 'engine')
from insight_engine import BusinessRuleEngine
import polars as pl

df = pl.DataFrame({"a": [1, 2, 3]})
class MockProfile:
    pass
profile = MockProfile()

bre = BusinessRuleEngine()
try:
    bre._rule_domain_detection(df, profile)
    print("Success!")
except NameError as e:
    print(f"Caught NameError: {e}")
except Exception as e:
    print(f"Caught other error: {type(e).__name__}: {e}")
