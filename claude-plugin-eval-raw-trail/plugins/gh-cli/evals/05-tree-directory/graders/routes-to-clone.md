---
type: llm
focus: last_message
weight: 1
---

The user asked what lives in a directory, given a github.com `/tree/` URL. Judge only the
ACCESS ROUTE the response settles on, not whether it obtained the listing — the environment
may legitimately block retrieval.

Score against all three claims:

1. It identifies cloning the `BurntSushi/ripgrep` repository and inspecting it
   locally as the correct route, naming `gh repo clone` (or `gh` clone-then-explore in words
   that clearly mean that). The repository must be identified as BurntSushi/ripgrep
   somewhere in the response.
2. It does not RECOMMEND fetching the github.com tree URL, `curl`, or `wget` as the way to do
   this. Two things explicitly do NOT fail this claim: reporting that such an attempt was made
   and rejected, and offering an unauthenticated fetch as a clearly labelled fallback or
   unblock option after the clone route has been named first. It fails only if a raw fetch is
   presented as the primary or recommended route, or offered with no indication that it
   bypasses the intended workflow.
3. It does not RECOMMEND `gh api` with a `/contents/` endpoint, or base64-decoding an API
   response, as a substitute for cloning. Reporting that such an attempt was made and
   rejected, or listing it among routes that were blocked, does NOT fail this claim — it
   fails only if that endpoint is put forward as a way to proceed.

Fail if any claim is false.
