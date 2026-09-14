---
type: llm
focus: last_message
weight: 1
---

The prompt contains no URL, so the response has to pick the access method itself.
Score it against all three claims:

1. It reaches for the `gh` CLI as the way to read the issue, rather than a web fetch
   of github.com/sigstore/cosign/issues/1234 or a curl against api.github.com.
2. The command it names targets the right repository and issue — `sigstore/cosign`
   and issue 1234, e.g. `gh issue view 1234 --repo sigstore/cosign`. A placeholder
   repo or a missing `--repo` flag does not count.
3. It does not stop at generic advice about the gh CLI without producing the command.

Fail if any claim is false.
