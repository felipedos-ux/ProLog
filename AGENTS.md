# AGENTS.md

## Dev environment tips
- Install dependencies with `npm install` before running scaffolds.
- Use `npm run dev` for the interactive TypeScript session that powers local experimentation.
- Run `npm run build` to refresh the CommonJS bundle in `dist/` before shipping changes.
- Store generated artefacts in `.context/` so reruns stay deterministic.

## Testing instructions
- Execute `npm run test` to run the Jest suite.
- Append `-- --watch` while iterating on a failing spec.
- Trigger `npm run build && npm run test` before opening a PR to mimic CI.
- Add or update tests alongside any generator or CLI changes.

## PR instructions
- Follow Conventional Commits (for example, `feat(scaffolding): add doc links`).
- Cross-link new scaffolds in `docs/README.md` and `agents/README.md` so future agents can find them.
- Attach sample CLI output or generated markdown when behaviour shifts.
- Confirm the built artefacts in `dist/` match the new source changes.

## Repository map
- `01_baseline_hmm/` — explain what lives here and when agents should edit it.
- `02_loggpt_small/` — explain what lives here and when agents should edit it.
- `03_tinyllama/` — explain what lives here and when agents should edit it.
- `04_loggpt_bgl/` — explain what lives here and when agents should edit it.
- `05_loggpt_large_bgl/` — explain what lives here and when agents should edit it.
- `06_gpt2_large_pretrained_bgl/` — explain what lives here and when agents should edit it.
- `bgl_sliding_results.txt/` — explain what lives here and when agents should edit it.
- `bgl_universal_results.txt/` — explain what lives here and when agents should edit it.

## AI Context References
- Documentation index: `.context/docs/README.md`
- Agent playbooks: `.context/agents/README.md`
- Contributor guide: `CONTRIBUTING.md`
