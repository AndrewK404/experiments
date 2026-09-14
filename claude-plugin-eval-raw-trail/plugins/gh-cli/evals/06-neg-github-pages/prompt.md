---
max_turns: 8
timeout_seconds: 180
allowed_tools: [Read, Glob, Grep, Skill, WebFetch, "Bash(gh:*)", "Bash(curl:*)", "Bash(git:*)"]
runs: 3
---

Fetch https://kubernetes.github.io/ingress-nginx/ and tell me what the default backend does.
