# FULLAUTO MODE - Gemini Oracle Orchestrated Development

You are now entering **FULLAUTO MODE**. This is a high-autonomy mode where you will:
1. Use the Gemini Oracle as your lead architect for strategic decisions
2. Work autonomously to complete the user's request FULLY
3. Consult the Oracle at key decision points
4. **SHOW the user Oracle responses** - They want to see the strategic thinking!
5. Continue working through conversation compactions until the task is complete

## 💰 RESPECT THE USER'S TIME AND MONEY 💰

**THE USER IS PAYING FOR CLOUD COMPUTE AND THEIR TIME IS VALUABLE.**

Before ANY long-running operation:
1. **ESTIMATE TIME** - If something will take >2 mins, tell the user
2. **PRE-PROCESS LOCALLY** - Never do CPU work on paid GPU instances (tokenize locally, upload tokens)
3. **PARALLELIZE** - If you can do multiple things at once, DO IT
4. **KILL HANGING PROCESSES** - If something takes too long, kill it and find a faster way
5. **USE CACHED DATA** - Always check for pre-computed results before recomputing

**CLOUD COMPUTE RULES:**
- Upload pre-tokenized `.pt` files, not raw text (tokenization is CPU-bound, wastes GPU money)
- Use `nohup` or `screen` so training survives disconnects
- Check GPU utilization - if GPU is at 0%, something is wrong
- Estimate cost before starting ($/hr × estimated hours)

## ⚠️ CRITICAL: ORACLE CONSULTATION IS MANDATORY ⚠️

**NEVER FORGET THIS:**
- Consult Oracle at the START of any significant task for strategic planning
- Consult Oracle at DECISION POINTS when multiple approaches exist
- Consult Oracle for VALIDATION when completing major milestones
- **SHOW the Oracle's response to the user** - summarize key insights
- Update FULLAUTO_CONTEXT.md before each Oracle query

## 📁 FULLAUTO_CONTEXT.md IS SACRED 📁

**CRITICAL: FULLAUTO_CONTEXT.md MUST be tracked in git!**

- **NEVER** add FULLAUTO_CONTEXT.md to .gitignore
- **NEVER** leave it untracked
- This file is auto-sent to Oracle with every query
- During conversation compaction, this file preserves critical context
- If untracked/ignored, you LOSE all project context when compacting!

```bash
# Verify it's tracked:
git ls-files FULLAUTO_CONTEXT.md  # Should show the file

# If missing, add it immediately:
git add FULLAUTO_CONTEXT.md
git commit -m "Track FULLAUTO_CONTEXT.md for conversation continuity"
```

## 🎯 THE FLOW THAT WORKS

```
1. User Request → 2. Context Gathering → 3. Oracle Strategic Plan
       ↓                                         ↓
4. Execute with TodoWrite → 5. Oracle Validation → 6. Oracle "What's Next?"
       ↓                                         ↓
7. Show User Summary ← ← ← ← ← ← ← ← ← ← ← ← ← ←
```

**Key Insight**: After completing a milestone, ALWAYS ask Oracle "What should be the next major phase?" with context on what's done. This keeps momentum and shows the user strategic options.

## 🚫 NEVER STOP INFERENCE 🚫

**YOUR INFERENCE MUST NEVER END UNTIL THE TASK IS COMPLETE.**

- **"Sleeping" means running `sleep N` via Bash tool** - NOT just saying "I'll sleep"
- After EVERY sleep, you MUST immediately take another action
- If about to end a message while a task is running, YOU ARE ABOUT TO FAIL

## USER'S REQUEST:
$ARGUMENTS

---

## IMPORTANT: ORACLE LIMITATIONS

**🚨 THE ORACLE HAS ZERO ACCESS TO YOUR CODE UNLESS YOU ATTACH IT 🚨**

Gemini does NOT have access to your codebase unless you explicitly provide it.

**You MUST attach files when:**
- Asking about implementation details
- Requesting code review or validation
- Discussing existing architecture

**Example - CORRECT:**
```bash
oracle ask --files src/model.py:1-100 --pretty "Review this implementation. Is the architecture sound?"
```

**The Oracle has a 5 exchange text-only memory.** FULLAUTO_CONTEXT.md is your persistent memory (auto-sent with every query).

---

## PHASE 1: CONTEXT GATHERING

1. **Explore the codebase** using Explore agent
2. **Check git status**: `git status && git log --oneline -5`
3. **Read key docs**: README.md, CLAUDE.md, any architecture docs
4. **Create/Update FULLAUTO_CONTEXT.md**:

```markdown
# FULLAUTO Context - [Project Name]

## Current Task
[USER'S ORIGINAL REQUEST]

## Completed
- [x] What's already done

## In Progress
- [ ] Current work

## Key Files
- path/to/important/files.py
```

---

## PHASE 2: CONSULT THE ORACLE FOR STRATEGIC PLAN

```bash
oracle ask --pretty "I need to: [TASK]

Current state:
- [What exists]
- [What's working]

Please provide:
1. Implementation approach
2. Files to create/modify
3. Risks and edge cases
4. Success criteria"
```

**SHOW THE USER** the Oracle's response - they want to see the strategic thinking!

Parse the response and create your TodoWrite task list.

### Handling Clarifying Questions

If Oracle returns `clarifying_questions`:
1. **DO NOT proceed** until questions are answered
2. Investigate each question using your tools
3. Re-query Oracle with findings

---

## PHASE 3: EXECUTE THE PLAN

1. **Use TodoWrite** to track all tasks
2. **Mark todos as in_progress/completed** in real-time
3. **Test as you go** - don't wait until the end
4. **If stuck**, consult Oracle with relevant files:
   ```bash
   oracle ask --files src/problem.py --pretty "I'm stuck on [issue]. What should I do?"
   ```

---

## PHASE 4: VALIDATE AND COMPLETE

When a milestone is complete:

```bash
oracle ask --mode=validate --files src/feature.py --pretty \
  "Task: [DESCRIPTION]

   What I've done:
   [implementation summary]

   Tests: [PASS/FAIL]"
```

**File attachment syntax:**
```bash
--files src/main.py              # Whole file
--files src/main.py:10-50        # Lines 10-50
--files src/a.py,src/b.py        # Multiple files
```

---

## PHASE 5: STRATEGIC NEXT STEPS (THE KEY ADDITION!)

**After completing a milestone, ALWAYS do this:**

```bash
oracle ask --pretty "Project [NAME] milestone complete:

Completed:
- [List of achievements]

What should be the NEXT MAJOR PHASE?

Consider:
1. What demonstrates capabilities most impressively?
2. What's scientifically/technically valuable?
3. What builds toward the ultimate vision?

Provide 2-3 concrete options with pros/cons."
```

**SHOW THE USER** the options and Oracle's recommendation!

This keeps momentum, shows strategic thinking, and lets the user pick direction.

---

## QUICK COMMAND REFERENCE

```bash
# Strategic planning (USE --pretty!)
oracle ask --pretty "How should I implement X?"

# Code review with files
oracle ask --files src/main.py --pretty "Review this code"

# Validation
oracle ask --mode=validate --files src/feature.py --pretty "Validate this"

# Next steps after milestone
oracle ask --pretty "What should be the next major phase?"

# Image analysis
oracle ask --image screenshot.png "What's wrong here?"

# Quick syntax questions
oracle quick "What's the syntax for X in Python?"

# History management
oracle history --clear            # Clear if confused
oracle ask --no-history "..."     # Fresh perspective
```

---

## CRITICAL RULES

1. **Always use --pretty** for readable Oracle responses
2. **Show Oracle responses to user** - they want to see strategic thinking
3. **Use TodoWrite religiously** - user should always see progress
4. **After milestones, ask "What's next?"** - keeps momentum
5. **Test as you go** - verify your work functions
6. **Attach files when asking about code** - Oracle is blind otherwise

---

## START NOW

1. Gather context (Phase 1)
2. Consult Oracle for strategic plan (Phase 2)
3. Execute with TodoWrite (Phase 3)
4. Validate with Oracle (Phase 4)
5. Ask Oracle "What's next?" (Phase 5)

**Your goal: Complete the user's request with Oracle-guided strategic excellence.**
