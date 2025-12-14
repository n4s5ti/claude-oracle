# Oracle CLI Changelog

## v2.1.0 - 2025-12-14: Natural Language Parser Update

### Summary
Major usability improvement: The oracle CLI now accepts natural language without requiring quotes or underscores. Multiple files can be specified using `-f` flag multiple times or comma-separated.

---

### Changes to `oracle.py`

#### 1. Query Argument - Natural Language Support
**Before:**
```bash
# Had to use quotes or underscores
oracle ask "How should I implement caching?"
oracle ask review_my_code_for_bugs
```

**After:**
```bash
# Just type naturally!
oracle ask How should I implement caching?
oracle ask Review my code for bugs
```

**Implementation:**
- Changed `query` from single positional argument to `nargs='*'`
- Added code to join query words: `query = ' '.join(args.query)`
- Applied same fix to `quick` and `imagine` commands

#### 2. Files Argument - Multiple File Support
**Before:**
```bash
# Comma-separated in single string (error-prone)
oracle ask --files "src/a.py,src/b.py" review
```

**After:**
```bash
# Multiple -f flags
oracle ask -f src/a.py -f src/b.py Review both files

# OR comma-separated
oracle ask --files src/a.py,src/b.py Review both files
```

**Implementation:**
- Changed `--files` from `nargs='+'` to `action='append'`
- Added `-f` as shorthand alias
- Added code to flatten and split comma-separated entries

#### 3. Code Changes (oracle.py)

**Lines 1986-2006 (ask parser):**
```python
# Before
ask_parser.add_argument("query", help="...")
ask_parser.add_argument("--files", help="Comma-separated...")

# After
ask_parser.add_argument("query", nargs='*', help="...multiple words without quotes")
ask_parser.add_argument("--files", "-f", action='append', help="...use -f multiple times OR comma-separated")
```

**Lines 2104-2120 (argument processing):**
```python
# Join query words
query = ' '.join(args.query) if args.query else ""

# Flatten files from action='append' and split commas
files = None
if args.files:
    files = []
    for f in args.files:
        for part in f.split(','):
            if part.strip():
                files.append(part.strip())
```

---

### Changes to Slash Commands

#### `/oracle` Command (`~/.claude/commands/oracle.md`)

**Features:**
- Clear command examples with natural language
- **AUDIT MODE** instructions for Claude
- Validation mode emphasis
- Multiple file syntax examples

**Key Sections:**
```markdown
## Commands Reference
# Strategic questions - just type naturally!
oracle ask How should I implement caching for this API?

# Multiple files (use -f multiple times OR comma-separated)
oracle ask -f src/a.py -f src/b.py -f src/c.py Review the architecture
oracle ask --files src/a.py,src/b.py,src/c.py Review the architecture

## AUDIT MODE
When user says "/oracle --audit", Claude MUST:
1. Identify ALL relevant code modified in session
2. Gather files using Glob/Read tools
3. Send ALL to Oracle for comprehensive audit
```

#### `/fullauto` Command (`~/.claude/commands/fullauto.md`)

**Features:**
- Mandatory Oracle validation after code changes
- Simple oracle usage examples
- Validation reminder section

**Key Sections:**
```markdown
## MANDATORY ORACLE VALIDATION AFTER CODE CHANGES
EVERY TIME you write or modify code, you MUST validate with Oracle:
oracle ask --mode=validate --files path/to/changed/file.py Validate my implementation

## ORACLE USAGE (SIMPLE!)
# All of these work!
oracle ask How should I implement caching?
oracle ask --files src/main.py Review this code for bugs
oracle ask -f src/a.py -f src/b.py Review the architecture
```

---

### Testing

Created `test_parser.py` to verify argument parsing:

```
Multi-word query no quotes
  Input: oracle ask This is a test
  OK Query: This is a test
  OK Files: (none)

Single file then query
  Input: oracle ask --files src/a.py Review this code
  OK Query: Review this code
  OK Files: ['src/a.py']

Multiple files with -f
  Input: oracle ask -f src/a.py -f src/b.py Review code
  OK Query: Review code
  OK Files: ['src/a.py', 'src/b.py']

Comma-separated files
  Input: oracle ask --files a.py,b.py,c.py Review all
  OK Query: Review all
  OK Files: ['a.py', 'b.py', 'c.py']

ALL TESTS PASSED!
```

---

### Migration Notes

- **Backwards compatible**: Quoted strings still work
- **Shorthand**: `-f` is alias for `--files`
- **Installation**: Copy updated `oracle.py` to `~/.oracle/oracle.py`

---

### Files Modified

1. `/mnt/c/Users/nira/Documents/Code/claude-oracle/oracle.py` - CLI parser
2. `~/.oracle/oracle.py` - Installed version
3. `~/.claude/commands/oracle.md` - /oracle slash command
4. `~/.claude/commands/fullauto.md` - /fullauto slash command
5. `/mnt/c/Users/nira/Documents/Code/claude-oracle/test_parser.py` - Test script
