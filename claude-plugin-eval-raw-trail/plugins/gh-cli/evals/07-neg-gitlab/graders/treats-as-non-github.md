---
type: llm
focus: last_message
weight: 1
---

This is a GitLab issue URL, not GitHub. The GitHub CLI cannot read it. This is a
should-NOT-fire case: the plugin must leave it alone.

Score against both claims:

1. It does NOT suggest `gh` in any form as the way to read this issue, and does NOT
   describe the URL as a GitHub resource.
2. It either summarizes the issue, or reports a genuine network or access failure against
   gitlab.com. Being unable to reach the network is acceptable and is NOT a failure of this
   grader; routing a GitLab URL through the GitHub CLI is.

Fail if either claim is false.
