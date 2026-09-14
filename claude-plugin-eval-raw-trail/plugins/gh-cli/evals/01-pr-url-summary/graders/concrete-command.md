---
type: llm
focus: last_message
weight: 1
---

Score this response against all three claims:

1. It names a single, runnable `gh` command for pull request 9876 in the `cli/cli`
   repository — either `gh pr view 9876 --repo cli/cli` or `gh pr view` against the
   full PR URL. A command with a placeholder like `owner/repo` instead of `cli/cli`
   does not count.
2. It does NOT stop at generic advice such as "use the gh CLI" or "the gh CLI is
   authenticated" without producing the actual command and its arguments.
3. It does NOT suggest `curl`, `wget`, or a web fetch of the PR URL as the way to
   get the answer.

Fail if any claim is false.
