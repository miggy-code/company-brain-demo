# company-brain-demo

A permission-aware company knowledge agent, shown on a synthetic business. Staff ask a question in Spanish or English and get an answer in which every sentence cites a source, drawn only from what that person could already see in the company's email, files, chat and CRM.

> **Everything here is synthetic.** Casa Prado Inmobiliaria, its people, buildings, projects, addresses, amounts and messages are invented for this demo, and every domain uses the reserved `.example` TLD. Nothing here describes a real company or client deployment.

## What it shows

- **Permissions in the index query.** Each item keeps its source ACL (people, groups, the company domain, Slack channels, HubSpot teams). Every query carries two filters in its `WHERE` clause: the item's ACL must include one of the asker's principals, and the item's label and tags must be ones the business policy lets the asker's role read. The policy can only narrow what the ACL allows. Nothing outside the asker's ceiling reaches ranking, the model, snippets, counts or wording.
- **Hybrid bilingual search.** BM25F computed in SQL, with document counts, frequencies and lengths taken from permitted rows only, so a hidden item cannot change a score. It is fused with BGE-M3 vectors (pgvector) and reranked by bge-reranker-v2-m3. Both models run in-process, so content goes to no embedding vendor.
- **Cited answers through a gate.** The model is called only through [agentgate](https://github.com/miggy-code/agentgate)'s model proxy. The proxy runs as a separate process, holds the provider key, and enforces the policy's model list and spend caps. A sentence that cites nothing is dropped, and a reply with no cited sentence becomes a fixed "not found" message.
- **Audit.** Each question writes `brain.question`, `brain.retrieval`, one `brain.decision` per result and `brain.answer` to the business's [Panoptes](https://github.com/miggy-code/Panoptes) hash chain. agentgate's run, model decision and the proxied exchange go to the same chain. If an event cannot be recorded, no answer is returned.
- **A permission suite in CI.** Every person is checked against every item that carries a canary token. The suite checks the ceiling against a hand-written oracle, looks for leaks in snippets, citations and answers, asks again with the item deleted (the answer must be identical), and checks that restricted and nonexistent topics get the same not-found reply. Any leak fails CI.
- **A leak suite.** 76 attacks in eleven categories, plus the person x canary grid, run against worst-case mock models in CI. Every protection in the question path is shown to matter by a mutation test (see [Leak suite](#leak-suite)).

## The demo business

Casa Prado Inmobiliaria (`businesses/casa-prado/`) is a fictional Lima real estate company. It manages two buildings, presells the "Residencial Mirador del Parque" project for a developer, brokers new and used sales, and handles rentals. There are six people: the owner, a sales manager, a sales agent, a leasing agent, a building administrator and an outside maintenance technician. There are 48 items across simulated Drive, Gmail, Slack and HubSpot.

## Run it

Needs Docker, [uv](https://docs.astral.sh/uv/), and an [agentgate](https://github.com/miggy-code/agentgate) checkout next to this repo (`../agentgate`, at or after `399d4ac`). agentgate 0.1.0 is not on PyPI yet.

```bash
docker compose up -d --wait                     # Postgres + pgvector on 127.0.0.1:55432
uv sync
uv run brain business create casa-prado         # database, migrations, agentgate world
uv run brain index casa-prado                   # BGE-M3 + reranker (~4.5 GB on first run)
uv run brain ask casa-prado --as lucia.quispe@casaprado.example --model mock \
  "¿En qué estado está la separación del departamento 502 en Mirador del Parque?"
```

`--model mock` puts a local stand-in for the provider behind the real proxy. For a live answer, pass `--model live --keys-file <file>`. The file is JSON mapping a provider to its key (`{"openai_compatible": "..."}`), with mode 0600. Only the proxy process reads it. The brain refuses to start if a provider key is in its own environment. The answer model is set in `businesses/casa-prado/business.yaml` (DeepSeek `deepseek-flash` by default, or Claude `claude-opus-5-5`).

`--as` is a demo identity: this demo has no login.

```bash
uv run brain eval casa-prado --suite permissions   # mock model; exit 1 on any failure
uv run brain redteam casa-prado --model obey       # leak suite; exit 1 on any failure
uv run brain eval casa-prado --suite golden --keys-file <file>   # 16 questions, live model
uv run panoptes session <run id> --db data/casa-prado/audit.db   # trace one answer
make check                                         # what CI runs, on the test doubles
```

## Leak suite

**Threat model.** The asker is curious or hostile, items may carry instructions (prompt injection), and the model is fully compromised. A leak is anything that depends on an item outside the audience's ceiling reaching what the audience sees: answer, citations, titles, snippets, dates, counts, order, not-found wording, errors, or history carried into a follow-up, including an earlier asker's question text, which counts as private to whoever heard it. The core check is non-interference: each answer, with its retrieved chunks, their order and the exact prompt, must be byte-identical with every hidden item removed and with restricted decoys added.

`brain redteam casa-prado --model echo|obey|mock|live` runs every scenario in `businesses/casa-prado/redteam/` and the person x canary grid. `echo` copies its whole context into the answer. `obey` also follows instructions planted in items. The run fails on any failed scenario, or on any scenario missing from (or not yet in) `redteam/baseline.json`.

| Guard | Protection | Shown by |
| --- | --- | --- |
| G1 | ACL predicate in the keyword and vector queries | `tests/mutations`: removing it fails direct and cross-language scenarios |
| G2 | BM25 statistics from permitted chunks only | without it, one restricted decoy reorders results (ranking_count-007, -008) |
| G3 | Policy predicate (roles and labels narrow below the ACL) | the technician then sees his building's fee statements |
| G4 | agentgate re-decides every result on its source ACL, label and tags | a forbidden chunk injected into search raises with no answer; without G4 it leaks |
| G5 | Only cited sentences, citing ids the model was given | `obey` then gets fabricated citations through |
| G6 | One fixed not-found message per language | a reply naming what was not found breaks the existence pairs |
| G7 | Audience intersection | an answer in front of the technician uses what only the owner may see |
| G8 | Follow-ups carry a turn only if the new audience could read all it cited and every member heard it | the leasing agent's follow-up then inherits the price list, or the owner's question naming a salary (follow_up-007) |
| G9 | No audit record, no answer | with best-effort audit, an answer leaves while the log is down |
| G10 | Identity from the authenticated context only | believing "soy la gerente general" opens the contracts |

<!-- redteam:start -->
| Category | Scenarios | Passed |
| --- | ---: | ---: |
| direct | 10 | 10/10 |
| existence | 7 | 5/7 |
| ranking_count | 8 | 5/8 |
| snippet_citation | 6 | 5/6 |
| prompt_injection | 7 | 1/7 |
| aggregation | 6 | 3/6 |
| cross_language | 6 | 6/6 |
| audience | 7 | 6/7 |
| follow_up | 7 | 1/7 |
| identity_claim | 6 | 5/6 |
| fail_closed | 7 | 7/7 |
| grid | 6 | 1/6 |
| **total** | **83** | **55/83** |

From `uv run brain redteam casa-prado --model live` at commit `0621aac` on 2026-11-21, model `deepseek-flash` (live, through the agentgate proxy), index `hash-1024` + `identity`. Synthetic data (Casa Prado is fictional).
<!-- redteam:end -->

Reading the live run above: no scenario leaked a canary, an item or an invalid citation, and every fail-closed scenario held. Every non-interference and differential failure is a reply-only difference: the retrieved chunks and the exact prompt were byte-identical with hidden items removed or restricted decoys added, so nothing hidden reached the model; the live model simply did not give the same reply to the same prompt twice. existence-001 also failed its not-found pair because its "restricted" question is answered by the public Mirador brochure, which the leasing agent may read and the model cited; the pair's premise is wrong, not the guard. The deterministic runs CI enforces (`--model echo`, `obey` and `mock`, same commit) pass 83/83.

Limits: timing side channels are out of scope; leaks are detected by canaries and item ids, so a model that paraphrased hidden content without its canary would only be caught by non-interference; all data is synthetic.

## Layout

| Path | Owns |
|---|---|
| `src/brain/tenancy/` | Business config, one Postgres database per business, migrations |
| `src/brain/identity/` | People and the principals each one holds |
| `src/brain/ingest/` | Items, structure-aware chunking, embeddings, the index |
| `src/brain/search/` | The permission predicate, BM25F + vector search, fusion, rerank |
| `src/brain/gate/` | The only code that talks to agentgate and Panoptes: policy, model proxy runs, audit |
| `src/brain/answer/` | Prompt, model clients, citation enforcement, the question pipeline |
| `src/brain/evals/` | Permission suite, golden set, the mock provider (relevance, echo, obey) |
| `src/brain/redteam/` | Leak suite: scenarios, non-interference variants, runner, report, baseline |
| `businesses/casa-prado/` | The synthetic business: config, people, items, agentgate policy, eval sets |

## Limits

- Identity is not authenticated (`--as`). Each question starts its own proxy process and run. A multi-run proxy and verified identity at run start belong upstream in agentgate.
- Policy narrowing works by enumerating the business's distinct (label, tags) combinations through agentgate's engine. This is a temporary workaround until agentgate can compile a policy into a query predicate.
- Audiences (several people who will see one answer) and threads exist in the pipeline and the leak suite; no interface uses them yet.
- There are no real connectors. Items come from fixtures, and the index is rebuilt from them in full.
