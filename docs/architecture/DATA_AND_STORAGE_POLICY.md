# V4 data and storage policy

Updated: 2026-07-14

The V4 runtime separates source code, immutable runtime objects, mutable query
projections, external scientific data, models/vendor runtimes, and credentials.
This boundary is enforced by `RuntimePaths`, `ArtifactStore`, and `RunIndex`.

| Class | Config | Default | Git policy |
| --- | --- | --- | --- |
| Run compatibility views | `AUTOPLANNER_RUNS_ROOT` | `results/.autoplanner/runs` | ignored |
| Immutable CAS | `AUTOPLANNER_ARTIFACT_STORE_ROOT` | `results/.autoplanner/artifacts` | ignored |
| Rebuildable SQLite index | `AUTOPLANNER_RUN_INDEX_PATH` | `results/.autoplanner/run_index.sqlite3` | ignored |
| Runtime caches | `AUTOPLANNER_CACHE_ROOT` | `results/.autoplanner/cache` | ignored |
| Hash-bound source documents | `AUTOPLANNER_SOURCE_ROOT` | `results/.autoplanner/sources` | ignored |
| External datasets | `AUTOPLANNER_EXTERNAL_DATA_ROOT` | `data_external` | ignored |
| Model weights | `AUTOPLANNER_MODEL_ROOT` | `<external-data>/models` | ignored |
| Vendor runtimes | `AUTOPLANNER_VENDOR_ROOT` | `vendor` | ignored |
| Credentials | environment / OS secret manager | none | never stored |

## Authority boundaries

- ArtifactStore proves byte identity and deduplicates content. It grants no
  chemistry, evidence, stock, or route-completion authority.
- RunIndex accelerates queries. It can be deleted and rebuilt from immutable
  manifests without changing any scientific artifact.
- Run compatibility files are mutable projections for old readers. Editing one
  does not edit the CAS object it was materialized from.
- Resolver cache entries are namespaced by parser authority and service
  endpoints. Successful entries are reused; failed entries expire and can only
  cause conservative rejection.
- Source documents require their own digest/provenance binding. Merely placing
  HTML or a PDF in the source root does not make it trusted evidence.

## Credential policy

New V4 components resolve Codex credentials in this order:

1. `AUTOPLANNER_CODEX_API_KEY` injected by the process or OS secret manager;
2. `OPENAI_API_KEY` injected by the process;
3. an explicitly configured `AUTOPLANNER_CODEX_KEY_PATH` outside the repository.

There is deliberately no implicit `<repository>/key.txt` fallback. Existing
legacy launchers retain compatibility temporarily and must be migrated in P7/P9.
Credentials, source values, and key paths must not appear in metrics, prompts,
run manifests, artifact metadata, logs, or exceptions.

## Retention and garbage collection

- Run manifest pointers automatically pin their manifest object.
- Scientific artifacts referenced by retained manifests must be supplied as
  explicit GC pins until transitive manifest pin discovery lands.
- Explicit Program admission indexes its historical graph and projection refs
  with `shadow_program_admission_*` authority scopes.  This pins their bytes
  for GC but grants no proof, route, or completion authority; the immutable
  run-local admission event remains the replay binding.  GC also replays these
  events directly, so recreating an empty RunIndex cannot orphan Program refs.
- GC is a dry run by default and reports candidate digests, ages, and bytes.
- Deletion requires explicit confirmation and revalidates every candidate
  digest immediately before removing it.
- Index/database files are not CAS objects and may be rebuilt at any time.

## Migration boundary

The repository still contains legacy hard-coded `results/shared`, `vendor`,
model, and `key.txt` defaults. They are compatibility debt, not V4 defaults.
The V4 `RunKernel`, Campaign Director, workers, benchmark, and unified CLI must
use this policy; P7/P9 will measure and remove the remaining legacy fallbacks.
