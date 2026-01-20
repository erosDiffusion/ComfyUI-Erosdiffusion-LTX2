---
name: code-reviewer
description: Monitor codebase changes and generate standardized analysis documentation
---

# Code Reviewer and Documenter Skill

## Purpose

Continuously monitor a codebase for changes and produce standardized per-commit documentation including architecture changes, features, regressions, and breaking changes.

## Trigger

This skill should be invoked:
1. When user requests codebase analysis
2. Periodically to check for new commits
3. After any development session

## Quick Check Command

```powershell
# Check for new commits since last analyzed
git log --oneline HEAD ^LAST_ANALYZED_SHA
```

## Output Location

All documentation goes into `bot-analysis/` folder in the repository root.

## Document Structure

```
bot-analysis/
+-- INDEX.md              # Main navigation (chronological commit table)
+-- README.md             # Quick start and status
+-- commits/
    +-- 01_XXXXXXX.md     # First commit analysis
    +-- 02_XXXXXXX.md     # Second commit analysis
    +-- ...
    +-- NN_XXXXXXX.md     # Latest commit (HEAD)
```

## Per-Commit File Template

```markdown
# Commit XXXXXXX - [Short Description]

| Field | Value |
|-------|-------|
| SHA | `XXXXXXX` |
| Timestamp | YYYY-MM-DD HH:MM:SS +ZZZZ |
| Message | [commit message] |
| Sequence | #N of M |
| Delta | +X minutes from previous |
| AI Model | [model name or Unknown] |

**Navigation:** [Index](../INDEX.md) | Prev: [XXXXXX](NN_XXXXXX.md) | Next: [XXXXXX](NN_XXXXXX.md)

---

## Status

| Aspect | State |
|--------|-------|
| Video | [Working/Broken/etc] |
| Audio | [Working/Broken/etc] |
| GUI | [Working/Broken/etc] |
| Tests | [X passing] |

---

## Architecture Changes

[Diff tree or "No changes"]

---

## Features Introduced

[New features or "None"]

---

## Breaking Changes

[Breaking changes or "None"]

---

## Regressions

[New regressions introduced or fixed, with ID like R-XXX]

---

## Notes

[Additional observations]
```

## Continuous Monitoring Workflow

### Step 1: Check for New Commits

```powershell
git log --format="%h|%ai|%s" HEAD -1
```

Compare with last entry in `commits/` folder.

### Step 2: If New Commits Found

1. Get commit details:
   ```powershell
   git show COMMIT_SHA --stat
   git diff PREV_SHA..COMMIT_SHA --name-status
   ```

2. Create new commit file: `commits/NN_XXXXXXX.md`

3. Update `INDEX.md` with new row

4. Update navigation links in previous commit file

### Step 3: Analyze for Breaking Changes

Check for:
- Signature changes in exported functions/classes
- Removed or renamed files
- Changed input/output types
- Modified API contracts

### Step 4: Update Regression Tracking

- Assign ID (R-XXX) to new regressions
- Link to commit that introduced them
- Update status when fixed

## Key Principles

1. **Objectivity:** Reference SHA commits, not subjective descriptions
2. **Time-Driven:** Order by timestamp, from initial to latest
3. **Navigatable:** Use Prev/Next/Index links
4. **Delta Focus:** Emphasize what changed, not cumulative state
5. **Breaking Changes:** Always highlight API/interface changes
6. **Non-Coding:** Analyze only, do not modify source code

## Regression ID Format

```
R-XXX: [Description]
Introduced: SHA (timestamp)
Fixed: SHA or UNRESOLVED
Impact: [affected components]
```

## Status Values

| Status | Meaning |
|--------|---------|
| Working | Functional |
| Broken | Known regression |
| WIP | Work in progress |
| Unknown | Not tested |
