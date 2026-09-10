# Bugs found while building the hpcp test suite

Do **not** fix these in `hpcp.py` as part of the test-suite work. Report here; product fixes are a separate change.

Format per entry:

## BUG-N: short title

- **Symbol:** `function_or_area`
- **Repro:** steps / minimal snippet
- **Observed:** ...
- **Expected (per README/docs/intent):** ...
- **Suite handling:** `xfail` / `skip` / asserted current behavior (say which)
