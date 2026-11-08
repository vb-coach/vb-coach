## Summary

<!-- What changes and why, in one paragraph. -->

## Testing

<!-- Paste the commands you ran and their output. "Tests pass" on its own is not evidence. -->

```
uv run pytest -q
```

## Risk

<!-- High, medium or low, using the path tiers in CLAUDE.md. Use the highest tier the PR touches. -->

Tier:

<!-- Which simulated outputs, scores or demos change, and by roughly how much. Write "none" if none do. -->

Affected outputs:

<!-- If an existing test assertion changed: was the old expectation wrong, and how do you know? -->

## How to revert

<!-- Usually `git revert <merge commit>`. Note anything that a revert would not undo. -->
