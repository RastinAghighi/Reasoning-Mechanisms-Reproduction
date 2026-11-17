# Experiment Log

Chronological record of implementation decisions, failures, fixes, and results.
Newest entries at the bottom. Each entry should be short and point to the run
directory (or `results/summaries/` file) that backs it.

**Entry template**

```markdown
### YYYY-MM-DD — <short title>
- **Experiment:** 01 | 02 | infra
- **Type:** decision | failure | fix | result
- **Run:** results/<experiment>/<run_id> (or n/a)
- **Details:** what was done and why.
- **Outcome / next step:**
```

For results, state explicitly whether a number is a **reproduction**
(and the paper value it is compared against) or an **extension**.

---

### 2026-11-17 — Repository foundation
- **Experiment:** infra
- **Type:** decision
- **Run:** n/a
- **Details:** Set up `src/qmind_reasoning` (YAML config with `extends` and
  `--set` overrides, strict schema validation, run directories with
  provenance metadata, seeding), per-experiment entry points and READMEs,
  and lightweight tests. Heavy ML dependencies are an optional extra
  (`.[experiments]`) so that the foundation installs and tests without them.
- **Outcome / next step:** Experiments not started. Open items before
  Experiment 1: verify SPARE-PRM checkpoint ID, pin model/dataset revisions,
  record the target paper numbers.
