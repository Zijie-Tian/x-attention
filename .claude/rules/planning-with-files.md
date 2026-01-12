# Planning-with-Files Rule (MANDATORY)

**This rule MUST be followed when using the `planning-with-files` skill.**

## Overwrite Policy

When creating a new plan using `planning-with-files`:

1. **ALWAYS delete or overwrite** the previous plan files before creating new ones
2. Plan files are **NOT cumulative** - each new planning session starts fresh
3. The following files should be overwritten/recreated each time:
   - `task_plan.md` - The main task plan
   - `findings.md` - Research findings and discoveries
   - `progress.md` - Progress tracking

## Rationale

- Prevents stale plan data from mixing with new plans
- Keeps the planning context clean and focused on the current task
- Avoids confusion between different planning sessions

## Implementation

Before writing any new plan content:
```bash
# Remove existing plan files if they exist
rm -f task_plan.md findings.md progress.md
```

Or simply overwrite them completely with new content (Write tool will overwrite by default).
