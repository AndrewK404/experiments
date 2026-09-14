---
max_turns: 15
timeout_seconds: 300
allowed_tools: [Read, Glob, Grep, Skill, WebFetch, "Bash(gh:*)", "Bash(curl:*)", "Bash(git:*)"]
runs: 3
---

Use curl to hit https://api.github.com/repos/rust-lang/rust/releases and give me the three most recent release tags.
