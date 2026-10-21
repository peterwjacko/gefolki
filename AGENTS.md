# AGENTS.md

Dense optical-flow coregistration of remote sensing images: a Python library and the `gefolki` command line tool. Read `README.md` first. The default branch is `master`.

## Project rules

- Gate: `uv run ruff check . && uv run ruff format --check . && uv run pytest -m "not gpu"`. Run it before each push. CI (`.github/workflows/ci.yml`) runs it on each PR.
- Branch: `<type>/<issue>-<slug>`, for example `fix/14-merge-policy`.
- PR body: `Closes #<issue>`.
- Merge: `gh pr merge <n> --merge` (a merge commit; no squash, no rebase).
- Who merges: Peter. An agent merges its own PR only when all CI checks are green, and Peter approved merges in chat or in the /goal text.
