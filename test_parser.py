#!/usr/bin/env python3
"""Test the argument parser changes."""
import argparse

# Recreate the parser
parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(dest="command")

# ask command
ask_parser = subparsers.add_parser("ask")
ask_parser.add_argument("query", nargs='*')
ask_parser.add_argument("--files", "-f", action='append')
ask_parser.add_argument("--mode", default="plan")
ask_parser.add_argument("--pretty", action="store_true")

def process_files(args_files):
    """Flatten and split comma-separated files."""
    if not args_files:
        return None
    files = []
    for f in args_files:
        for part in f.split(','):
            part = part.strip()
            if part:
                files.append(part)
    return files

# Test cases
test_cases = [
    # (description, args)
    ("Multi-word query no quotes", ["ask", "This", "is", "a", "test"]),
    ("Single file then query", ["ask", "--files", "src/a.py", "Review", "this", "code"]),
    ("Multiple files with -f", ["ask", "-f", "src/a.py", "-f", "src/b.py", "Review", "code"]),
    ("Comma-separated files", ["ask", "--files", "a.py,b.py,c.py", "Review", "all"]),
    ("Mode and file", ["ask", "--mode=validate", "--files", "test.py", "Validate", "my", "work"]),
    ("Pretty flag", ["ask", "--pretty", "How", "should", "I", "fix", "this"]),
    ("Mixed: -f and comma", ["ask", "-f", "a.py,b.py", "-f", "c.py", "Check", "these"]),
]

print("=" * 60)
print("STRESS TESTING ORACLE ARGUMENT PARSER")
print("=" * 60)

all_passed = True
for desc, args in test_cases:
    print(f"\n{desc}")
    print(f"  Input: oracle {' '.join(args)}")
    try:
        parsed = parser.parse_args(args)
        query = ' '.join(parsed.query) if parsed.query else "(empty)"
        files = process_files(parsed.files) if parsed.files else "(none)"
        print(f"  OK Query: {query}")
        print(f"  OK Files: {files}")
        if query == "(empty)":
            print(f"  WARN: Query is empty!")
            all_passed = False
    except Exception as e:
        print(f"  FAIL ERROR: {e}")
        all_passed = False

print("\n" + "=" * 60)
if all_passed:
    print("ALL TESTS PASSED!")
else:
    print("SOME TESTS HAD ISSUES")
print("=" * 60)
