---
description: Analyze the latest commit using the code-reviewer skill
---

# Analyze Commit Workflow

Invoke this workflow after making commits to document changes in `bot-analysis/`.

## Steps

1. Get the latest commit SHA and compare with last analyzed:
```powershell
git log --oneline HEAD -1
```

2. Check the last analyzed commit in `bot-analysis/commits/` folder - identify the highest numbered file.

3. If there are new commits since last analysis:
   - Read the SKILL.md at `.agent/skills/code-reviewer/SKILL.md`
   - Follow the skill instructions to create new commit analysis files
   - Get commit details with:
     ```powershell
     git show HEAD --stat
     git diff HEAD~1..HEAD --name-status
     ```

4. Create a new commit file following the template in SKILL.md:
   - File name: `NN_XXXXXXX.md` where NN is sequence number and XXXXXXX is short SHA
   - Include AI Model field with current model being used
   - Update navigation links in previous commit file

5. Update `bot-analysis/INDEX.md` with the new commit row.

6. Report summary of what was documented.
