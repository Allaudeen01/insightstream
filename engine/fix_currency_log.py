# Run this from the insightstream root directory:
# python engine/fix_currency_log.py

import re

path = "engine/report_generator.py"

with open(path, encoding="utf-8") as f:
    content = f.read()

# Count occurrences before
before = content.count("[Currency Fix]")
print(f"Found {before} '[Currency Fix]' occurrences")

# Remove the 3-line block
pattern = r'\s*# ── FIX 2: Post-process all Paragraph elements to fix currency symbols ──\s*\n\s*log\.info\("\[Currency Fix\][^"]*"\)\s*\n\s*# elements = self\._fix_currency_symbols\(elements\)'
new_content = re.sub(pattern, '', content)

after = new_content.count("[Currency Fix]")
print(f"After removal: {after} '[Currency Fix]' occurrences")

if before != after:
    with open(path, "w", encoding="utf-8") as f:
        f.write(new_content)
    print("SUCCESS - block removed")
else:
    print("Pattern not matched - block may have different whitespace")
    # Show lines containing Currency Fix
    for i, line in enumerate(content.splitlines(), 1):
        if "Currency Fix" in line:
            print(f"  Line {i}: {line}")
