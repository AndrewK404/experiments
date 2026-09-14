# claude-plugin-eval-raw-trail

A raw first run of `claude plugin eval`, the harness built into Claude Code for measuring whether a
plugin actually adds anything over the bare model. The plugin under test is
[gh-cli](https://github.com/trailofbits/skills/tree/main/plugins/gh-cli) by Trail of Bits — it
intercepts GitHub URL fetches and reroutes Claude to the authenticated `gh` CLI. The 7-case suite
under `plugins/gh-cli/evals/` is mine, not the authors'.

## The whole point

```sh
cd plugins/gh-cli                 # the PLUGIN root, not the repo root

claude plugin eval init           # interview -> writes evals/
# ... 3 gates: inputs -> grader calibration -> cost

claude plugin eval . \            # full run
  --ablation with-without \       # no delta without this
  --allow-tools WebFetch Bash \   # the skill does not grant them itself
  --judge-model sonnet \          # haiku is too weak to judge these graders
  -j 3

# -> evals/results/<ts>/{aggregate-result.json, report.html}
# -> one number that matters: mean delta
```

The target (`.`) must come **before** `--allow-tools`: the flag is variadic and will eat the
positional argument if it comes after.

## Result

7 cases x 3 runs x 2 arms = 42 agent runs. 6.5 min, $7.83, Claude Code 2.1.270, sonnet judge.

| | with plugin | without | delta |
| --- | --- | --- | --- |
| mean score | 0.87 | 0.31 | **+0.56** |

| case | with | without | delta |
| --- | --- | --- | --- |
| 01-pr-url-summary | 0.83 | 0.00 | +0.83 |
| 02-raw-file-content | 0.78 | 0.00 | +0.78 |
| 03-explicit-curl-api | 0.83 | 0.00 | +0.83 |
| 04-issue-no-url | 1.00 | 0.17 | +0.83 |
| 05-tree-directory | 0.78 | 0.00 | +0.78 |
| 06-neg-github-pages *(should NOT fire)* | 1.00 | 1.00 | 0.00 |
| 07-neg-gitlab *(should NOT fire)* | 0.83 | 1.00 | -0.17 |

Reading it:

- Without the plugin the model never reaches for `gh` on its own — four of the five positive cases
  score a flat 0. That is the uplift.
- Case 07 is the one real finding: in 1 run of 3 the plugin arm treated a GitLab URL as GitHub.
  A should-NOT-fire case is what makes over-triggering visible at all.
- `skill-fired` (`tool_used: Skill`) is a trigger indicator, not a score: under ablation the harness
  keeps it out of both arms, so it never moves the delta. The two negative cases carry none — there
  the plugin firing at all is the failure, and `no-gh-suggestion` catches it.

## Layout

```
claude-plugin-eval-raw-trail/
├── README.md                                   <- this file
└── plugins/
    └── gh-cli/                                 <- the plugin under test, from upstream
        ├── README.md                           <- trimmed, plus the eval notes
        ├── .claude-plugin/plugin.json          <- name, version 1.6.0, author
        ├── skills/gh-cli/
        │   ├── SKILL.md                        <- what the eval actually measures
        │   ├── ...
        ├── hooks/                              <- the deterministic half of the plugin
        │   ├── hooks.json                      <- binds the hooks to PreToolUse / SessionEnd
        │   ├── ...
        └── evals/
            ├── 01-pr-url-summary/
            │   ├── prompt.md                   <- runs, max_turns, timeout, allowed_tools
            │   └── graders/
            │       ├── concrete-command.md     <- llm, w1
            │       ├── names-pr-view.md        <- regex, w1
            │       └── skill-fired.md          <- tool_used: Skill, display-only
            ├── 02-raw-file-content/            <- routes-to-clone, clones-repo (w0.5), skill-fired
            ├── ...
            └── results/2026-09-14T14-44-42-212Z/
                ├── aggregate-result.json       <- per-case, per-run, per-grader scores
                └── report.html                 <- same run, self-contained HTML
```

Each case is a `prompt.md` plus a `graders/` dir; 01 is shown in full, 02-07 list only their
grader names. Every case pairs an llm grader for the access route with a regex for the command,
plus `skill-fired` on the positives. Weights are 1 unless marked.

Plugin source: [trailofbits/skills](https://github.com/trailofbits/skills) @ `ce9ae2e`, CC BY-SA 4.0, README trimmed.
