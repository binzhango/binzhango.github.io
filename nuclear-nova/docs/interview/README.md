# LLM interview question bank

The article remains at `/posts/2025/10-30-llm-interview/`. Its original blog entry supplies publishing metadata and the introduction; `src/content/interview/` supplies the topic-organized questions. Read and Practice reuse the same server-rendered Markdown. Answers are not bundled into the client controller.

## Content and attribution

- [Reconciliation manifest](reconciliation.json): all 765 input entries map to 762 permanent question IDs. Three equivalent entries were merged; distinct constraints and follow-ups remain.
- [Similarity review](similarity-candidates.json): reviewed lexical candidates and merge/retain reasons.
- [Original snapshots](sources/) and [attribution notice](NOTICE.md): retained independently of Downloads.
- [Verification report](VERIFICATION.md): release checks and their scope.

## Editing questions

Each Markdown filename matches its permanent `questionId`. Preserve that ID when changing wording or editorial order. Metadata declares `topic`, optional `subtopic`, unique positive `order`, `kind`, `sources`, `status`, and `reviewedOn`. Keep the three authored sections in order:

```markdown
## Question
Standalone prompt.

## Brief answer
Concise answer, solution outline, suggested approach, or response framework.

## Detailed answer
Self-contained explanation, assumptions, worked examples, and limitations. Keep research references in frontmatter `sources`, not in the published question or answer body.
```

The collection schema and production build reject invalid topics, duplicate IDs/orders/aliases, missing sections, and unreviewed records. The 13 topics and eight legacy anchor aliases are defined in `src/utils/interview-topics.ts`. For a merge, preserve meaningful follow-ups, add the retired ID to `aliases`, and update the reconciliation manifest. Research technical changes against primary sources before marking them reviewed.

`scripts/interview/migrate.py` is the historical import utility, not an incremental editor: rerunning it would overwrite later editorial decisions. Edit canonical Markdown and the manifest directly instead.

## Local verification

Run from `nuclear-nova/`:

```sh
rtk proxy npm test
rtk proxy npx astro check
rtk proxy npm run build
rtk proxy npm run preview -- --host 127.0.0.1 --port 4324
```

With the preview running, run the browser suite in a separate terminal:

```sh
rtk proxy env INTERVIEW_URL=http://127.0.0.1:4324 ./node_modules/.bin/playwright test --config scripts/interview/playwright.config.ts
```

Screenshots and interaction measurements are written to the ignored `test-results/interview/` directory. Answer-specific executable checks and their runtime requirements are documented in the linked editorial reports. The temporary Python environment used during implementation is not a site runtime dependency.

Progress uses only `bz:interview:progress:v1` in local storage. Ratings are self-assessment. Resetting this bank does not clear other site storage. The initial view always remains Read, including after a reload or when JavaScript is unavailable.
