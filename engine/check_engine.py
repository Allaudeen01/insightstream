import sys
sys.path.insert(0, 'engine')
import insight_engine
print(f"detect_domain in insight_engine: {'detect_domain' in dir(insight_engine)}")
try:
    print(f"detect_domain source: {insight_engine.detect_domain.__code__.co_firstlineno}")
except Exception as e:
    print(f"Error getting detect_domain: {e}")

from insight_engine import BusinessRuleEngine
try:
    import inspect
    print(f"_rule_domain_detection source: {inspect.getsourcelines(BusinessRuleEngine._rule_domain_detection)[1]}")
except Exception as e:
    print(f"Error getting _rule_domain_detection: {e}")
