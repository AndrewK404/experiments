# gh-cli

Intercepts GitHub URL fetches and redirects Claude to the authenticated `gh` CLI. Vendored from
[trailofbits/skills](https://github.com/trailofbits/skills) — README trimmed and merged with the
eval notes, the rest of the plugin unchanged.

## Problem

`WebFetch`, MCP fetch tools and `curl`/`wget` don't carry GitHub auth: private repos 404,
unauthenticated API calls cap at 60/hour instead of 5,000, some responses come back incomplete.

## What it does

- **PreToolUse hooks** catch GitHub URLs in `WebFetch`, MCP fetch/scrape/crawl/extract tools and
  `curl`/`wget`, and answer with the right command — `gh pr view`, `gh issue view`,
  `gh release list`, `gh api`, or `gh repo clone` + Read for blob/tree/raw/contents URLs.
- **A `gh` PATH shim** blocks reading files via `gh api .../contents/...` and clones outside a
  session-scoped temp dir; **a SessionEnd hook** deletes those clones when the session ends.
- Non-GitHub domains, GitHub Pages, plain `git`, and greps that merely mention a URL pass through.

Needs [`gh`](https://cli.github.com/) installed and authenticated; without it the hooks stay silent.

## Evals

Ablation suite in `evals/`: every case runs twice, with the plugin and without, so the delta
measures what the skill adds over an unassisted model.

```sh
claude plugin eval . --ablation with-without --allow-tools WebFetch Bash --judge-model sonnet -j 3
```

| case | what it measures | should fire |
| --- | --- | --- |
| 01-pr-url-summary | PR URL -> concrete `gh pr view 9876 --repo cli/cli`, not generic advice | yes |
| 02-raw-file-content | `raw.githubusercontent.com` -> `gh repo clone` + local read | yes |
| 03-explicit-curl-api | an explicit `curl` against `api.github.com` gets substituted | yes |
| 04-issue-no-url | no URL in the prompt; the right repo and issue still land in the command | yes |
| 05-tree-directory | `/tree/` URL -> clone and inspect locally | yes |
| 06-neg-github-pages | a Pages docs site is not a repo — read it, suggest no `gh` | no |
| 07-neg-gitlab | a GitLab URL must not be routed through the GitHub CLI | no |

llm graders judge the **access route** the answer settles on, not whether the fetch succeeded — the
sandbox may have no network, and that must not score as a failure. `skill-fired` (`tool_used:
Skill`) is a trigger indicator only; under ablation it stays out of the score in both arms. The
sonnet judge and `--allow-tools WebFetch Bash` are both required: haiku misses the "recommended vs.
reported as blocked" distinction, and the skill grants itself no tools.

Recorded run: `evals/results/2026-09-14T14-44-42-212Z/`, numbers in the top-level README.
