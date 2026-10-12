# Exact disk values, refusals before hashing, single probes (group B2 of the 9.67 follow-ups)

## Problem

The disk-patch path (`-dd -diff`, and applying a disk patch to a disk) and the `-dd` copy mode share helpers that
read and write partition and filesystem identity. Several of these read values wrongly, so the wrong values get
written to every node or copy:

- **Filesystem labels.** `_probe_identity` parses `blkid -p -o export`, which prints bytes above 127 in `M-`
  notation in every locale (`röot` reads as `rM-CM-6ot`; probe 2026-10-11, util-linux on this host). Disk patches
  then write the mangled label to every node, and verify passes. The `-dd` path's `get_partition_details` parses
  cached `blkid -o export` with `line.split('=')[1]`, which cuts a label at its first `=`. Its `'TYPE' in line` test
  also matches `PTTYPE`, `SEC_TYPE` and `PART_ENTRY_TYPE`.
- **GPT partition names.** `get_partition_details` reads `Partition name: '...'` with `line.split("'")[1].strip()`.
  That cuts a name at an apostrophe and drops leading and trailing spaces. `sgdisk` prints the name raw between
  quotes, without escaping a quote inside it (probe: `'it's a  name '`). Disk patches write the truncated name to
  node GPTs when a name changes.
- **exFAT serials.** `_fs_identity_commands` passes the serial in blkid's form (`ABCD-1234`). `exfatlabel -i` rejects
  that with "invalid serial number" and accepts `0xABCD1234`. So exFAT UUID changes have never worked, in disk
  patches or in `-dd` UUID mirroring. Found by B1; its test is an expected failure.
- **Device paths of partitions.** `get_target_partition` (`'/dev/' + lsblk name`), `get_partitions` and
  `get_largest_partition` (`os.path.join(os.path.dirname(disk), lsblk name)`) build paths from kernel names.
  - A cciss disk's `cciss!c0d0p1` becomes a non-existent path; the real node is `/dev/cciss/c0d0p1`.
  - A disk given as `/dev/disk/by-id/...` makes the last two build the partition path inside `/dev/disk/by-id/`.
  - `_partition_devices` already maps `!` to `/` (9.67).
- **Refusals that come after long hashing passes.**
  - In `apply_disk_patch`, every raw area in the patch is hashed in full, and its payload hashed, before most of
    the refusals run: `Disk in use` for identity changes on mounted partitions, `Identity change error` (unsupported
    without `-fa`), B1's exclusive-open check, and "cannot be mounted read-write". Raw areas can be gigabytes.
  - In `process_get_disk_diff_image`, "raw area and its size changed" is checked inside the per-partition loop,
    after earlier partitions' file diffs. The `-dis` "a raw area must fit in one patch volume" refusal comes after all
    hashing.
- **Repeated probes.**
  - `apply_disk_patch` probes the partition-table type, then `read_disk_layout` probes it again.
  - `read_disk_layout` settles udev, then `_settle_partitions` settles again.
  - Each partition gets a cached `blkid` (in `get_partition_details`) plus a separate `blkid -p` (`_probe_identity`).
  - The verify read after identity changes re-runs the mkfs-parameter probes (`probe_fs_params`: `dumpe2fs`,
    `xfs_db` and similar), though verify compares only identity values.

## Goal

The disk tools read and write identity values exactly, find partitions on cciss-style disks and through by-id
paths, refuse before hashing whenever a refusal does not need a hash, and run each probe once per layout read. This
holds for disk patches and for `-dd` copy mode. It ships with the next release: 9.70, unless the user publishes 9.69
first, in which case B2 becomes part of it.

## Decisions (made with the user)

1. B2 fixes the shared helpers, so `-dd` copy mode gets the same fixes; tests cover both paths.
2. Design sections 1 and 2 below were presented and approved as written here.

## Design

### Part 1 — exact values (shared helpers)

1. **Filesystem type, UUID and label: one shared reader.**
   - It runs `blkid -p -o udev DEVICE` and takes `ID_FS_TYPE`, `ID_FS_UUID_ENC` and `ID_FS_LABEL_ENC`.
   - It decodes each `\xNN` escape to its byte, then decodes the bytes as UTF-8 with `surrogateescape`, so a byte
     that is not UTF-8 survives.
   - A missing value is `''`. blkid exits 2 when it finds nothing (BIOS boot and raw partitions): that is an answer,
     not an error, as today.
   - `_probe_identity` returns this reader's values.
   - `get_partition_details` uses it instead of its `blkid -o export` parse. Its `fs_type`, `fs_uuid` and
     `fs_label` then match `_probe_identity` exactly, and its fs-params probe uses that `fs_type`.
   - Labels like `röot =x` round-trip exactly.
2. **GPT names.**
   - `get_partition_details` takes the text between the first and the last `'` on the `Partition name:` line, with no
     stripping. An empty name (`''`) stays `''`.
   - `sgdisk` stays the GPT reader, so the other GPT fields (type GUID, unique GUID, attribute flags, size) read
     exactly as before, and disk patches made by 9.67–9.69 still compare equal on them.
3. **exFAT serial.**
   - `_fs_identity_commands('exfat', device, uuid=...)` passes `'0x' + uuid.replace('-', '')`.
   - This fixes disk patches and `-dd` UUID mirroring.
   - The B1 identity-tool test's exFAT UUID case stops being an expected failure.
4. **Partition device paths.**
   - `get_target_partition`, `get_partitions` and `get_largest_partition` build `'/dev/' + name.replace('!', '/')`
     from the kernel (lsblk) name, as `_partition_devices` does.
   - That maps `cciss!c0d0p1` to `/dev/cciss/c0d0p1`, and keeps by-id inputs from producing paths inside
     `/dev/disk/by-id/`.

### Part 2 — refusals before hashing, probes once

1. **Disk apply.** Every refusal that does not need a content hash runs before the first raw-area hash. In order:
   - the layout checks (as now);
   - the in-use refusals (as now);
   - the read-only live-mount refusals;
   - the identity refusals, including `Identity change error` without `-fa`;
   - the exclusive-open check (B1);
   - each raw payload file present on a volume with the recorded size (its full hash stays later);
   - each filesystem section's partition mountable read-write (the mount itself happens here, as now, before
     planning).

   The exclusive-open check covers every raw partition with an entry in the patch, plus every partition whose
   UUID or label the patch sets. Whether a raw area is already done is only known after hashing. This matches the
   existing policy that an in-use raw area is refused without being read; an offline apply skips one that is
   already done.

   Then raw areas are hashed (node area and payload), then the sections are planned, then the prompt.
2. **Disk patch generation.**
   - "raw area and its size changed" is checked for every partition of kind `raw` on the new layout before anything
     is hashed. Partitions that fall back to raw because they cannot be mounted on one side are only known once
     mounting is tried, so they keep the check where it is.
   - The partitions of kind `raw` are hashed first, before any filesystem section is diffed (today they are hashed in
     partition order, between the sections' file diffs).
   - The `-dis` "a raw area must fit in one patch volume" refusal then runs right after those raw hashes. It runs
     whenever the changed raw areas alone force more than one volume and the largest of them does not fit one
     volume's usable space. Whether a raw area is in the patch at all is only known once its old and new contents
     are hashed, so this refusal cannot come earlier without refusing unchanged areas.
   - The existing check after all hashing stays, for splits that the sections' payload causes.
   - (Corrected after the design was approved: the approved text said this refusal could run from the layout alone,
     before any hashing. That would refuse a raw area that turns out to be unchanged.)
3. **One probe per layout read.**
   - `read_disk_layout(device, name=None, pttype=None, fs_params=True)`: `apply_disk_patch` passes the
     partition-table type it already probed, instead of probing twice.
   - Each layout read settles udev once (today it settles twice).
   - Each partition gets one `blkid -p` run, through Part 1's shared reader.
   - The verify read after identity changes passes `fs_params=False` and skips the mkfs-parameter probes.
   - Any `read_disk_layout` result used for `disk.tsv` keeps `fs_params=True`.

## Compatibility

- Patch formats are unchanged. `disk.tsv` already escapes every field and writes with `surrogateescape`, so
  exactly-read labels and names round-trip.
- Disk patches made by 9.67–9.69 for disks with non-ASCII labels, or GPT names with apostrophes or edge spaces,
  carry the old mangled or truncated values. Applying them behaves as it did then: the node's exactly-read value
  may now differ from the patch's recorded base, which gives the existing "node differs from base" warning.
  Regenerating such a patch fixes it. The README says so.
- `-dd` copy mode: labels, GPT names and exFAT UUIDs that were copied wrongly are now copied exactly. Ordinary
  values are unchanged.

## Testing

Real-filesystem tests run as root with loop devices. Scratch files go only under `tmp_path`, and nothing is left
mounted or attached.

**Probe safety rule:** any ad-hoc root probe goes in a script file run with `bash <file>`. `set -euo pipefail` typed
at the top of a shell command is ignored in this harness. Every path variable gets a `: "${VAR:?}"` guard and a
`/tmp/claude-0/` prefix check, and devices are only `/dev/loop*` backed by scratch images.

- **Exact values:**
  - a GPT image with:
    - an ext4 label `röot =x`;
    - GPT names `it's a  name ` and `Ünïcode ✓`;
    - an exFAT partition (when `mkfs.exfat` and `exfatlabel` exist).
  - A disk patch from that image to a copy with other labels, names and an exFAT serial applies, and the node ends
    with exactly the golden's values.
  - A `-dd` copy of the image ends with the same values (the existing dd round-trip helpers in
    `tests/test_dd_fs_params.py`).
  - Unit tests of the shared reader cover `\x20`, `\x5c`, raw UTF-8 and invalid UTF-8, of the GPT-name parse, and of
    the exFAT argv.
- **Device paths:** unit tests with fake `lsblk` output and a fake sysfs, for a cciss disk and a by-id path.
- **Refusals:** for each refusal in Part 2 items 1-2, a spy on `_hash_area` (and on file hashing for generation)
  shows the refusal fires with zero hashes.
- **Probes:** call counters on `blkid`, `udevadm settle` and `probe_fs_params` show one partition-table probe and one
  settle per layout read, one `blkid` per partition, and no `probe_fs_params` in the verify read.
- **Compatibility:** the existing disk round-trip and dd tests stay green with identical `disk.tsv` values for
  ASCII-only disks.

## Review focus

1. **Label decoding.** Every escape `blkid -o udev` can produce decodes back to the original bytes, including a
   backslash in a label (`\x5c`), a label that is not UTF-8, and an empty label.
2. **GPT name parsing.** It covers an empty name, a name that is all spaces, and a name with several apostrophes.
3. **Refusal reordering.** No refusal is lost or weakened; nothing is written or mounted before a refusal that
   used to come first.
4. **`-dd` copy mode.** Behaviour on ordinary ASCII labels and names is byte-for-byte unchanged.
5. **Settle removal.** It does not bring back the udev re-read race 9.67 fixed: a loop device attached, then probed
   at once.

## Out of scope (later rounds)

Round C:
- `-e` anchoring at partition roots for `-dd -diff`;
- input images in single-filesystem patches;
- payload re-check cost;
- the `/x/out` test guard;
- group A's and B1's loud or rare leftovers.
