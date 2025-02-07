# hpcp
A simple script that can issue multiple `cp -af` commands simultaneously on a local system.

Optimized for use in HPC scenarios and featuring auto-tuning for files-per-process.

Includes an adaptive progress bar for copying tasks from multiCMD.

Tested on a Lustre filesystem with 1.5 PB capacity running on 180 HDDs. Compared to using `tar`, **hpcp** reduced the time for tarball/image release from over 8 hours to under 10 minutes.

## Development status

Basic functionality (parallel copy) should be stable.

Imaging functionality (source/destination as `.img` files) will be extended with differential image support (differential backup). Imaging is only available on Linux—similar to `tar`, but uses disk images.

Block-image functionality is in **beta**. Only available on Linux. Possible use case: cloning a currently running OS without mounting `/` as read-only.

hpcp.bat available on github: simple old tk based GUI intended for basic windows functionality.

## Important Implementation Detail

By default, **hpcp** only checks:
1. The file’s relative path/name is identical.
2. The file mtime is identical.
3. The last `-hs --hash_size` bytes (defaults to `65536`) are identical.

Although in most cases these checks should confirm that both files are identical, in certain scenarios (like bit rot), corrupted files might not be detected. If you need to verify file integrity rather than perform a quick sync, it is recommended to use the `-fh --full_hash` option.

Setting `-hs --hash_size` to `0` disables hash checks entirely. This can be helpful on HDDs, as they usually have suboptimal seek performance. However, HDDs are also more prone to bit rot. If the operator can accept that risk, it is possible to rely solely on mtime checks for file comparison by setting `hash_size` to `0`. (Though on a single HDD, the standard `cp` command is already well-optimized.)

## Installation

```bash
pipx install hpcp
```
or
```bash
pip install hpcp
```

After installation, **hpcp** is available as `hpcp`. You can check its version and libraries via:
```bash
hpcp -V
```

It is recommended to install via **pip**, but **hpcp** can also function as a single script using Python’s default libraries.

**Note**:  
- Using `pip` will optionally install the hashing library [xxhash](https://github.com/Cyan4973/xxHash), which can reduce CPU usage for partial hashing and increase performance when using `-fh --full_hash`.  
- `pip` also installs [multiCMD](https://github.com/yufei-pan/multiCMD), used to issue commands and provide helper functions. If it is not available, `hpcp.py` will use its built-in multiCMD interface, which is more limited, has lower performance, and may have issues with files containing spaces. Please install **multiCMD** if possible.

## Disk Imaging Feature Note

Only available on Linux currently!

`-dd --disk_dump` mode differs from the standard Linux `dd` program. **hpcp** will try to mount the block device/image file to a temporary directory and perform a file-based copy to an identically-created image file specified with `-di --dest_image`. This functionality is implemented crudely and is still an **alpha** feature. It works on basic partition types (it does not work with LVM) with GPT partition tables and has been proven able to clone live running system disks to disk images, which can then be booted without issues.  
The created disk image can be resized using the `-ddr --dd_resize` option to the desired size. (This feature is provided so that you can shrink the raw size of the resulting image and provides some shrink capability for XFS.)  
For partitions that **hpcp** cannot create a separate unique mount point, **hpcp** will fall back to using the Linux program `dd` to clone the drive. Note that this can be risky and can lead to broken filesystems if the drive is actively being written to. (However, since you generally cannot mount that partition on the current OS, the real-world scenarios for this remain limited.)

## Remove Extra Feature Note

`-rme --remove_extra`: Especially when combined with `-rf`, **PLEASE PAY CLOSE ATTENTION TO YOUR TARGET DIRECTORY!**  
`--remove_extra` will remove **all** files that are not in the source path. When you are copying a file into a folder, you almost certainly do not want to use this!

## Remove Feature Note

`-rm --remove` can remove files in bulk. This might be helpful on distributed file systems like Lustre, as it only gathers the file list once and performs bulk deletion rather than the default recursive deletion in the Linux `rm` program.

`-rf --remove_force` implies `--remove`. **Use with care!** This skips the interactive check requiring user confirmation before removing. If **hpcp** did not generate the correct file list from the specified source paths, hopefully you have fast enough reflexes to press `Ctrl + C` repeatedly to stop all parallel deletion processes if you realize a mistake.

`-b --batch`: Using `-b` with `-rm` will gather the file list for all `source_paths` first, then issue the remove command. This can be helpful because **hpcp** will tune its `-f --files_per_job` parameter accordingly for each task, and running one large remove job might be faster than running many small ones. This is especially useful when working with glob patterns like `*`.

```bash
$ hpcp -h
usage: hpcp [-h] [-s] [-j MAX_WORKERS] [-b] [-v] [-do] [-nds] [-fh] [-hs HASH_SIZE]
               [-f FILES_PER_JOB] [-sfl SOURCE_FILE_LIST] [-fl TARGET_FILE_LIST] [-cfl]
               [-dfl [DIFF_FILE_LIST]] [-tdfl] [-nhfl] [-rm] [-rf] [-rme] [-e EXCLUDE]
               [-x EXCLUDE_FILE] [-nlt] [-V] [-pfl] [-si [SRC_IMAGE ...]]
               [-siff [LOAD_DIFF_IMAGE ...]] [-d DEST_PATH] [-di DEST_IMAGE]
               [-dis DEST_IMAGE_SIZE] [-diff] [-dd] [-ddr DD_RESIZE]
               [src_path ...]

Copy files from source to destination

positional arguments:
  src_path              Source Path

options:
  -h, --help            show this help message and exit
  -s, --single_thread   Use serial processing
  -j, -m, -t, --max_workers MAX_WORKERS
                        Max workers for parallel processing. Default is 4 * CPU count.
                        Use negative numbers to indicate {n} * CPU count, 0 means 1/2 CPU
                        count.
  -b, --batch           Batch mode, process all files in one go
  -v, --verbose         Verbose output
  -do, --directory_only
                        Only copy directory structure
  -nds, --no_directory_sync
                        Do not sync directory metadata, useful for verfication
  -fh, --full_hash      Checks the full hash of files
  -hs, --hash_size HASH_SIZE
                        Hash size in bytes, default is 65536
  -f, --files_per_job FILES_PER_JOB
                        Base number of files per job, will be adjusted dynamically.
                        Default is 1
  -sfl, -lfl, --source_file_list SOURCE_FILE_LIST
                        Load source file list from file. Will treat it raw meaning do not
                        expand files / folders files are seperated using newline. If
                        --compare_file_list is specified, it will be used as source for
                        compare
  -fl, -tfl, --target_file_list TARGET_FILE_LIST
                        Specify the file_list file to store list of files in src_path to.
                        If --compare_file_list is specified, it will be used as targets
                        for compare
  -cfl, --compare_file_list
                        Only compare file list. Use --file_list to specify a existing
                        file list or specify the dest_path to compare src_path with. When
                        not using with file_list, will compare hash.
  -dfl, --diff_file_list [DIFF_FILE_LIST]
                        Implies --compare_file_list, specify a file name to store the
                        diff file list to or omit the value to auto-determine.
  -tdfl, --tar_diff_file_list
                        Generate a tar compatible diff file list. ( update / new files
                        only )
  -nhfl, --no_hash_file_list
                        Do not append hash to file list
  -rm, --remove         Remove all files and folders specified in src_path
  -rf, --remove_force   Remove all files without prompt
  -rme, --remove_extra  Remove all files and folders in dest_path that are not in
                        src_path
  -e, --exclude EXCLUDE
                        Exclude source files matching the pattern
  -x, --exclude_file EXCLUDE_FILE
                        Exclude source files matching the pattern in the file
  -nlt, --no_link_tracking
                        Do not copy files that symlinks point to.
  -V, --version         show programs version number and exit
  -pfl, --parallel_file_listing
                        Use parallel processing for file listing
  -si, --src_image [SRC_IMAGE ...]
                        Source Image, mount the image and copy the files from it.
  -siff, --load_diff_image [LOAD_DIFF_IMAGE ...]
                        Not implemented. Load diff images and apply the changes to the
                        destination.
  -d, -C, --dest_path DEST_PATH
                        Destination Path
  -di, --dest_image DEST_IMAGE
                        Destination Image, create a image file and copy the files into
                        it.
  -dis, --dest_image_size DEST_IMAGE_SIZE
                        Not implemented. Destination Image Size, specify the size of the
                        destination image to split into. Default is 0 (No split).
  -diff, --get_diff_image
                        Not implemented. Compare the source and destination file list,
                        create a diff image of that will update the destination to
                        source.
  -dd, --disk_dump      Disk to Disk mirror, use this if you are backuping / deploying an
                        OS from / to a disk. Require 1 source, can be 1 src_path or 1 -si
                        src_image, require 1 -di dest_image.
  -ddr, --dd_resize DD_RESIZE
                        Resize the destination image to the specified size with dd
```