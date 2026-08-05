# Enabling CI

`github-actions-ci.yml` is the test workflow. It is parked here rather than in
`.github/workflows/` because the token used to create this repository lacks the
GitHub `workflow` OAuth scope, so pushing a workflow file is rejected.

To enable it:

```bash
gh auth refresh -h github.com -s workflow
mkdir -p .github/workflows
git mv docs/ci/github-actions-ci.yml .github/workflows/ci.yml
git commit -m "ci: enable GitHub Actions" && git push
```

Or paste the file into the repository through the GitHub web UI, which is not
subject to the same scope restriction.
