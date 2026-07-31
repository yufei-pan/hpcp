# Content-only copy mode (`-co` / `--content_only`)

Date: 2026-07-31  
Status: Approved design (pending implementation plan)

## Problem

`hpcp` always preserves file and directory metadata when copying:

- New file copies use `cp -af --sparse=always` (posix) or `shutil.copy2` (elsewhere).
- Identical-file short-circuit still runs `copystat`, `chown`, `chmod`, and `utime`.
- Directory sync creates dirs then applies `copystat`, `chown`, and `utime`.

There is no way to copy **content only** and leave destination mode, ownership, and timestamps alone. That matters when destinations should inherit parent ACL / setgid / setuid / umask defaults instead of mirroring the source.

Existing related flags are not enough:

- `-ncd` / `--no_create_dir` — skips creating missing destination directories entirely.
- `-nds` / `--no_directory_sync` — skips the directory metadata sync job (useful for verification), not a content-only file copy mode.

## Goal

Add an opt-in **content-only** mode that:

1. Copies file contents without applying source mode, owner, or timestamps.
2. Still **creates** missing destination directories (unlike `-ncd`).
3. Does **not** sync directory metadata, so new folders get default filesystem permissions and correctly inherit parent ACL / setgid / setuid behavior.
4. Leaves already-identical destinations completely untouched (no metadata touch-up).

Default behavior (no flag) remains full metadata preserve, unchanged.

## Non-goals

- Changing `-nds` or `-ncd` semantics.
- Partial preserve options (mode-only, times-only, owner-only).
- Reworking Windows `xcopy` attribute flags beyond using `shutil.copy` on the non-posix primary path.
- Version bump / PyPI release (can follow in a separate change).

## Approach

**Flag + skip metadata paths** (chosen over extracting a shared helper or post-copy stripping).

Wire a module global `CONTENT_ONLY` the same way as `NO_CREATE_DIR`, set from argparse in `main`, and branch at the existing copy / sync call sites.

Post-copy stripping was rejected: it still briefly applies source attrs, can fail on `chown`, and is wasteful. A shared metadata helper is nicer long-term but is unnecessary refactor for one flag in this single-file module.

## Behavior

| Path | Default (today) | With `-co` / `--content_only` |
|---|---|---|
| New file copy | `cp -af --sparse=always` / `copy2` | `cp -f --sparse=always` / `shutil.copy` — content only; dest attrs from umask / FS defaults |
| Fallback non-sparse file copy | `cp -af` | `cp -f` |
| Identical file | `copystat` + `chown` + `chmod` + `utime` | Leave dest untouched |
| New directory | `makedirs` + `copystat` + `chown` + `utime` | `makedirs` only — FS-default perms so ACL/setgid/setuid inherit from parent |
| Existing directory | Metadata overwrite | No metadata changes |

### Distinction from `-ncd`

| Flag | Creates missing dest dirs? | Syncs metadata? |
|---|---|---|
| (default) | Yes | Yes |
| `-co` | Yes | No |
| `-ncd` | No | N/A for missing dirs |
| `-co` + `-ncd` | No | No (where dirs already exist) |

## CLI

```text
-co, --content_only
    Content-only copy: do not sync mode, owner, or timestamps on files
    or directories. Still create missing destination directories using
    filesystem-default permissions (so parent ACL/setgid/setuid inherit).
```

- `action='store_true'`, default off.
- Orthogonal to `-ncd` and `-nds`; combinations are allowed.

## Implementation sketch

1. Add `CONTENT_ONLY = False` near other module globals (`NO_CREATE_DIR`).
2. Add argparse `-co` / `--content_only`; assign `CONTENT_ONLY = args.content_only` in `main` (same pattern as `no_create_dir`).
3. In `copy_file`:
   - When `CONTENT_ONLY` and files are identical: return without metadata sync.
   - When copying: use `cp -f --sparse=always` (and `cp -f` on non-sparse fallback) instead of `-af`; on non-posix use `shutil.copy(..., follow_symlinks=False)` instead of `copy2`.
4. In `sync_directory_metadata`:
   - Keep `makedirs` / existence handling.
   - Skip `copystat`, `chown`, and `utime` when `CONTENT_ONLY`.
5. Ensure any other preserve-on-copy paths in the same functions follow the same rule.
6. Document the flag in `README.md` help/flag listing if that section is maintained by hand.

Workers observe the flag via the existing global / process-init pattern used for other flags.

## Error handling

- Retry, backoff, and failure reporting stay unchanged.
- Under `-co`, permission/`chown` failures that would occur during metadata sync simply do not happen because those calls are skipped.
- Copy failures still surface the same way as today.

## Testing

No automated test suite in this repo. Verify manually (or with a small ad-hoc script):

1. **New file + `-co`:** content matches source; mode/owner/mtime are not forced from source (umask/FS defaults).
2. **New dir + `-co`:** directory is created; inherits parent defaults (e.g. setgid parent → child has setgid).
3. **Identical file re-run + `-co`:** destination metadata unchanged after the run.
4. **Default (no `-co`):** preserve behavior unchanged (`cp -af` / metadata sync still applied).
5. **`-co` still creates dirs** (contrast with `-ncd`, which does not).

## Success criteria

- `-co` copies content without applying source mode, owner, or timestamps anywhere (new copy, identical touch-up, directory sync).
- Missing destination directories are still created under `-co`.
- New directories inherit parent FS defaults (ACL / setgid / setuid) rather than source metadata.
- Default mode behavior is unchanged.
