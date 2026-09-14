---
max_turns: 25
timeout_seconds: 600
allowed_tools: [Read, Glob, Grep, Skill, WebFetch, "Bash(gh:*)", "Bash(curl:*)", "Bash(git:*)"]
runs: 3
---

Pull the contents of https://raw.githubusercontent.com/astral-sh/uv/main/README.md and tell me what the project does.
