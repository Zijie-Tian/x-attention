# GPU Allocation Policy (MANDATORY)

**This rule MUST be followed at the START of every session before running ANY program that uses GPU resources.**

## For Direct Execution (Claude Code itself)

1. **Before running the FIRST GPU-related command** in a session, you MUST ask the user which GPU(s) to use
2. Use `AskUserQuestion` tool to ask: "Which GPU(s) should I use for this session?"
3. **Only ask ONCE per session** - remember the allocation for all subsequent commands
4. Apply the GPU allocation using `CUDA_VISIBLE_DEVICES` environment variable in all commands

## For Agent/Subagent Execution (Task tool)

1. **Before spawning ANY agent** that may run GPU commands, you MUST ask the user for GPU allocation for that specific agent
2. Each agent needs its own GPU allocation question - do NOT reuse the main session's allocation
3. Include the GPU allocation in the agent's prompt/instructions
4. Format: "Agent [agent-name/type] needs GPU access. Which GPU(s) should this agent use?"

## What Counts as "GPU-related commands"

- Any Python script that imports `torch`, `tensorflow`, or similar ML libraries
- Docker commands that use `--gpus` flag
- Scripts in `scripts/`, `tools/`, `eval/` directories
- Benchmark or evaluation runs
- Model inference or training
- Any command involving CUDA operations

## Example GPU Allocation Question

```
Header: "GPU Selection"
Question: "Which GPU(s) should I use for [task description]?"
Options:
- "GPU 0" - Use only GPU 0
- "GPU 1" - Use only GPU 1
- "GPUs 0,1" - Use GPUs 0 and 1
- "GPUs 2,3" - Use GPUs 2 and 3
- "All GPUs" - Use all available GPUs
```

## Applying GPU Allocation

```bash
# In bash commands
CUDA_VISIBLE_DEVICES=0,1 python script.py

# In Docker commands
docker run --gpus '"device=0,1"' ...

# Or modify CUDA_VISIBLE_DEVICES in script variables
```

## Important Reminders

- **NEVER** assume GPU availability - always ask first
- **NEVER** run GPU commands without explicit user allocation
- **REMEMBER** the allocation throughout the session (for direct execution)
- **ASK SEPARATELY** for each agent that needs GPU access
