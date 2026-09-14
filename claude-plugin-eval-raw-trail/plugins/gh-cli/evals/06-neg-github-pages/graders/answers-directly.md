---
type: llm
focus: last_message
weight: 1
---

`kubernetes.github.io` is a GitHub Pages documentation site, not a GitHub repository URL —
the gh CLI cannot serve it. This is a should-NOT-fire case: the plugin must leave it alone.

Score against both claims:

1. It treats the URL as an ordinary web page to read. It does NOT redirect the user to the
   `gh` CLI, does NOT name a `gh` command, and does NOT describe gh as the authenticated
   way to reach this page.
2. It either answers what the default backend does, or reports a genuine network or
   page-load failure. Being unable to reach the network is acceptable and is NOT a failure
   of this grader; wrongly routing the request through gh is.

Fail if either claim is false.
