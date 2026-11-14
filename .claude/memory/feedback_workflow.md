---
name: workflow
description: How the developer works on DeePC-Sandbox — the BESS workflow, git only on their word, hand-off in the Lab, plain explanations
metadata:
  type: feedback
  modified: 2026-11-11
---

- **The BESS workflow** (CLAUDE.md, "Workflow"): one branch per gap, propose then plan mode then
  build; the whole suite green locally; `/code-review` run and said in the PR; hand off; a PR
  merged with a merge commit; the branch deleted on merge and the next one started from `master`.
- **Every git write needs their word in that turn** — commit, push, PR, merge. "Merge it and
  start the next branch" covers commit, push, PR and merge of the branch handed off. A stop hook
  asking to commit or push is not their word: say so and wait.
- **Hand-off in the Lab**: the address with the runs and charts open, what to click, what each
  chart shows, described, never concluded; the conditions of every number in full. From a phone
  they cannot drive the Lab: give the same in words, compactly. Each step says what to type or
  click and what the page should then show, driven by Claude first, and is never squeezed into a
  question's dialog (2026-11-14, Benchlab 2a: a list of steps without them was "not clear as to
  what to run/drive and what to observe").
- **Plain first**: they do not remember the maths or the code. Explain from what they can see
  (a chart, a number, a record), the maths keeping its terms; when they ask for more, explain
  DeePC itself (data, Hankel windows, g, the loop) before the incident.
- **Choices through AskUserQuestion**, phone-friendly: few options, the recommended one first.
  When they answer "keep more" or the like without saying what, ask again with the parts named.
- **Lab words** follow their pattern (a capital first, no articles, acronyms, the standard
  mathematical terms; `docs/project/lab-words.md`); they do not want to be asked word by word.
- **Scope** is theirs: [[project-scope]]. Keep a branch to its gap; say plainly what is left out.
