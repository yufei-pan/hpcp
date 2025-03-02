#!/usr/bin/env python3
# /// script
# requires-python = ">=3.6"
# dependencies = [
#     "argparse",
#     "xxhash",
#     "multiCMD>1.19",
# ]
# ///
import os
import sys
import time
import argparse
import multiprocessing
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import shutil
import glob
import re
import fnmatch
import functools
import tempfile
import select
import pathlib
import random
import re
import threading
from collections import deque
try:
	import multiCMD
	assert float(multiCMD.version) > 1.19
except:	
	import select
	import sys
	class multiCMD:
		version = 'build_in_0.09'
		def run_commands(commands,timeout = 0,max_threads=1,quiet=False,dry_run=False):
			rtnList = []
			for command in commands:
				if isinstance(command,str):
					command = command
				else:
					# escape the spaces in the command
					command = [arg.encode(errors='backslashreplace').decode('utf-8').replace(' ','\\ ') for arg in command]
					command = ' '.join(command)
				if not quiet:
					print('*' * 80)
					print('> ' , command)
				try:
					if not dry_run:
						rtnList.append(os.popen(command).read().strip().split('\n'))
					else:
						rtnList.append([f'Dry run: {command}'])
				except Exception as e:
					rtnList.append([f'Error: {e}'])
				if not quiet:
					print('\n'.join(rtnList[-1]))
					print('*' * 80)
			return rtnList
		def run_command(command,timeout=0,quiet=False,dry_run=False):
			return multiCMD.run_commands([command],timeout=timeout,quiet=quiet,dry_run=dry_run)[0]
		def genrate_progress_bar(iteration, total, prefix='', suffix='',columns=120):
			noPrefix = False
			noSuffix = False
			noPercent = False
			noBar = False
			# if total is 0, we don't want to divide by 0
			if total == 0:
				return f'{prefix} iteration:{iteration} total:{total} {suffix}\n'
			percent = f'|{("{0:.1f}").format(100 * (iteration / float(total)))}% '
			length = columns - len(prefix) - len(suffix) - len(percent) - 3
			if length <= 0:
				length = columns - len(prefix) - len(suffix) - 3
				noPercent = True
			if length <= 0:
				length = columns - len(suffix) - 3
				noPrefix = True
			if length <= 0:
				length = columns - 3
				noSuffix = True
			if length <= 0:
				return f'{prefix}\niteration:\n {iteration}\ntotal:\n {total}\n| {suffix}\n'
			if iteration == 0:
				noBar = True
			filled_length = int(length * iteration // total)
			progress_chars = '▁▂▃▄▅▆▇█'
			fractional_progress = (length * iteration / total) - filled_length
			char_index = int(fractional_progress * (len(progress_chars) - 1))
			bar_char = progress_chars[char_index]
			if filled_length == length:
				bar = progress_chars[-1] * length
			else:
				bar = progress_chars[-1] * filled_length + bar_char + '_' * (length - filled_length)
			lineOut = ''
			if not noPrefix:
				lineOut += prefix
			if not noBar:
				lineOut += f'{bar}'
				if not noPercent:
					lineOut += percent
			else:
				if length >= 16:
					lineOut += f' Calculating... '
			if not noSuffix:
				lineOut += suffix
			return lineOut
		def print_progress_bar(iteration, total, prefix='', suffix=''):
			prefix += ' |' if not prefix.endswith(' |') else ''
			suffix = f'| {suffix}' if not suffix.startswith('| ') else suffix
			try:
				columns, _ = multiCMD.get_terminal_size()

				sys.stdout.write(f'\r{multiCMD.genrate_progress_bar(iteration, total, prefix, suffix, columns)}')
				sys.stdout.flush()
				if iteration == total:
					print(file=sys.stdout)
			except:
				if iteration % 5 == 0:
					print(multiCMD.genrate_progress_bar(iteration, total, prefix, suffix))
		def get_terminal_size():
			try:
				_tsize = os.get_terminal_size()
			except:
				try:
					import fcntl, termios, struct
					packed = fcntl.ioctl(0, termios.TIOCGWINSZ, struct.pack('HHHH', 0, 0, 0, 0))
					_tsize = struct.unpack('HHHH', packed)[:2]
				except:
					import shutil
					_tsize = shutil.get_terminal_size(fallback=(120, 30))
			return _tsize
		def input_with_timeout_and_countdown(timeout, prompt='Please enter your selection'):
			"""
			Read an input from the user with a timeout and a countdown.

			Parameters:
			timeout (int): The timeout value in seconds.
			prompt (str): The prompt message to display to the user. Default is 'Please enter your selection'.

			Returns:
			str or None: The user input if received within the timeout, or None if no input is received.
			"""
			# Print the initial prompt with the countdown
			print(f"{prompt} [{timeout}s]: ", end='', flush=True)
			# Loop until the timeout
			for remaining in range(timeout, 0, -1):
				# If there is an input, return it
				if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
					return input().strip()
				# Print the remaining time
				print(f"\r{prompt} [{remaining}s]: ", end='', flush=True)
				# Wait a second
				time.sleep(1)
			# If there is no input, return None
			return None
		
try:
	import xxhash
	hasher = xxhash.xxh64()
	xxhash_available = True
except ImportError:
	import hashlib
	hasher = hashlib.blake2b()
	xxhash_available = False

version = '9.03'
__version__ = version

# ---- Helper Functions ----
class Adaptive_Progress_Bar:
	def __init__(self, total_count = 0, total_size = 0,refresh_interval = 0.1,last_num_job_for_stats = 5,custom_prefix = None,custom_suffix = None,process_word = 'Processed',use_print_thread = False):
		self.total_count = total_count
		self.total_size = total_size
		self.refresh_interval = refresh_interval
		self.item_counter = 0
		self.size_counter = 0
		self.scheduled_jobs = 0
		self.stop_flag = False
		self.process_word = process_word
		self.startTime = time.perf_counter()
		self.last_n_jobs = deque(maxlen=last_num_job_for_stats)
		self.custom_prefix = custom_prefix
		self.custom_suffix = custom_suffix
		self.lastCallArgs = None
		self.use_print_thread = use_print_thread
		if use_print_thread:
			...
			# Disabling print thread as for python threading and process pool coexistance bug
			self.use_print_thread = False
			# self.print_thread = threading.Thread(target=self.print_progress_thread)
			# self.print_thread.daemon = True
			# self.print_thread.start()
	def print_progress_thread(self):
		while not self.stop_flag:
			# sleep for refresh_interval
			time.sleep(self.refresh_interval)
			self.print_progress()
	def print_progress(self):
		if self.total_count == self.total_size == self.item_counter == self.size_counter == 0:
			return
		if self.total_count > 0 and self.item_counter >= self.total_count:
			self.stop_flag = True
		# job: (num_files, cpSize, cpTime, files_per_job)
		if not self.custom_prefix and not self.custom_suffix:
			job_count = len(self.last_n_jobs)
			if job_count > 0:
				last_n_files_sum, last_n_size_sum, last_n_time_sum , files_per_job_sum= [sum(x) for x in zip(*self.last_n_jobs)]
				files_per_job = files_per_job_sum / job_count
				last_n_time = last_n_time_sum / job_count
				last_n_file_speed = last_n_files_sum / last_n_time if last_n_time_sum > 0 else 0
				last_n_size_speed = last_n_size_sum / last_n_time if last_n_time_sum > 0 else 0
			total_time = time.perf_counter() - self.startTime
			total_file_speed = self.item_counter / total_time
			total_size_speed = self.size_counter / total_time
			if total_file_speed:
				remaining_time = (self.total_count - self.item_counter) / total_file_speed
			elif total_size_speed:
				remaining_time = (self.total_size - self.size_counter) / total_size_speed
			else:
				remaining_time = '∞'
		if self.custom_prefix:
			prefix = self.custom_prefix
		else:
			prefix = f'{format_bytes(self.item_counter,use_1024_bytes=False,to_str=True)}F ({format_bytes(self.size_counter)}B) {self.process_word} |'
			if self.scheduled_jobs:
				prefix += f' {self.scheduled_jobs} Scheduled'
			if job_count > 0:
				prefix += f' {files_per_job:0>4.1f} F/Job '
		if self.custom_suffix:
			suffix = self.custom_suffix
		else:
			suffix = f'{format_bytes(total_size_speed)}B/s {format_bytes(total_file_speed,use_1024_bytes=False,to_str=True)}F/s |'
			if job_count > 0:
				suffix += f' {last_n_time:.1f}s: {format_bytes(last_n_size_speed)}B/s {format_bytes(last_n_file_speed,use_1024_bytes=False,to_str=True)}F/s |'
			suffix += f' {format_time(remaining_time)}'
		callArgs = (self.item_counter, self.total_count, prefix, suffix)
		if callArgs != self.lastCallArgs:
			self.lastCallArgs = callArgs
			multiCMD.print_progress_bar(*callArgs)
	def stop(self):
		self.stop_flag = True
		if self.use_print_thread and self.print_thread.is_alive():
			self.print_thread.join()
	def update(self, num_files, cpSize, cpTime= 0 , files_per_job=1):
		self.item_counter += num_files
		self.size_counter += cpSize
		self.last_n_jobs.append((num_files, cpSize, cpTime, files_per_job))
		if not self.use_print_thread:
			self.print_progress()

_binPaths = {}
@functools.lru_cache(maxsize=None)
def check_path(program_name):
	#global __configs_from_file
	global _binPaths
	config_key = f'_{program_name}Path'
	program_path = (
		#__configs_from_file.get(config_key) or
		os.environ.get(program_name.upper() + '_PATH') or
		globals().get(config_key) or
		shutil.which(program_name)
	)
	if program_path:
		_binPaths[program_name] = program_path
		return True
	return False

_binCalled = ['lsblk', 'losetup', 'sgdisk', 'blkid', 'umount', 'mount','dd','cp', 'xcopy',
			  'fallocate','truncate', 
			  'mkfs', 'mkfs.btrfs', 'mkfs.xfs', 'mkfs.ntfs', 'mkfs.vfat', 'mkfs.exfat', 'mkfs.hfsplus', 
			  'mkudffs', 'mkfs.jfs', 'mkfs.reiserfs', 'newfs', 'mkfs.bfs', 'mkfs.minix', 'mkswap'
			  'e2fsck', 'btrfs', 'xfs_repair', 'ntfsfix', 'fsck.fat', 'fsck.exfat', 'fsck.hfsplus', 
			  'fsck.hfs', 'fsck.jfs', 'fsck.reiserfs', 'fsck.ufs', 'fsck.minix']
[check_path(program) for program in _binCalled]

def run_command_in_multicmd_with_path_check(command, timeout=0,max_threads=1,quiet=False,dry_run=False,strict=True):
	"""
	Run a command in multiCMD with path check.

	Args:
		command (str): The command to run.
		timeout (int, optional): The timeout value in seconds. Defaults to 0.
		max_threads (int, optional): The maximum number of threads to use. Defaults to 1.
		quiet (bool, optional): Whether to suppress the output. Defaults to False.
		dry_run (bool, optional): Whether to perform a dry run. Defaults to False.
		strict (bool, optional): Whether to exit if the command fails to find the bin. Defaults to True.

	Returns:
		list: The output of the command.
	"""
	global _binPaths
	# Check the path of the command
	if isinstance(command, str):
		command = command.split()
	if not command:
		print("Error: Command is empty.", file=sys.stderr, flush=True)
		sys.exit(1)
	if not isinstance(command[0],str):
		command[0] = str(command[0])
	if not command[0] in _binPaths:
		if not check_path(command[0]):
			print(f"Error: Command '{command[0]}' not found. Please consider installing it then retry.", file=sys.stderr, flush=True)
			if strict: sys.exit(127)
	# Run the command
	return multiCMD.run_commands([command], timeout=timeout, max_threads=max_threads, quiet=quiet, dry_run=dry_run)[0]

# -- Exclude --
def is_excluded(path, exclude=None):
	"""
	Check if a given path is excluded based on a list of patterns.

	Args:
		path (str): The path to check.
		exclude (list[str], optional): List of patterns to exclude. Defaults to None.

	Returns:
		bool: True if the path is excluded, False otherwise.
	"""
	if exclude is None:
		return False
	for pattern in exclude:
		if fnmatch.fnmatch(path, pattern):
			return True
	return False

def formatExclude(exclude = None,exclude_file = None) -> frozenset:
	if not exclude:
		exclude = set()
	else:
		exclude = set(exclude)
	if exclude_file:
		if os.path.exists(exclude_file):
			try:
				with open(exclude_file,'r') as f:
					exclude.update(f.read().splitlines())
			except:
				print(f"Error encounted while reading exclude file {exclude_file}, skipping")
		else:
			print(f"Exclude file {exclude_file} does not exist, skipping")

	exclude = set([re.sub(r'/+','/','*/'+ path if not path.startswith('*/') else path) if not path.startswith('/') else path for path in exclude ])
	# freeze frozenset
	return frozenset(exclude)

# -- DD --
def get_largest_partition(disk):
	"""
	Get the largest partition on the disk.

	Args:
		disk (str): The disk name or path.

	Returns:
		str: The path of the largest partition on the disk.
	"""
	partitions = run_command_in_multicmd_with_path_check(["lsblk", '-nbl', '-o', 'NAME,SIZE',disk])
	# sort by size
	partitions = sorted(partitions, key=lambda x: int(x.split()[1]))
	# Skip the first entry as it's the disk itself
	largest_partition = partitions[-2].split()[0] if len(partitions) > 1 else partitions[-1].split()[0]
	return os.path.join(os.path.dirname(disk), largest_partition)

def get_partitions(disk):
	"""Get all the partitions on the disk.

	Args:
		disk (str): The disk name or path.

	Returns:
		list: A list of partition paths.

	"""
	partitions = run_command_in_multicmd_with_path_check(["lsblk", '-nbl', '-o', 'NAME,SIZE',disk])
	# Skip the first entry as it's the disk itself
	partitions = [os.path.join(os.path.dirname(disk), partition.split()[0]) for partition in partitions[1:]]
	return partitions

def get_fs_type(path):
	"""
	Get the filesystem type of the path.

	Args:
		path (str): The path for which to determine the filesystem type.

	Returns:
		str: The filesystem type of the path.
	"""
	fs_type = run_command_in_multicmd_with_path_check(["lsblk", '-nbl', '-o', 'FSTYPE', path])[0].strip()
	return fs_type

def fix_fs(target_partition, fs_type=None):
	"""
	Fix the file system errors on the specified target partition.

	Args:
		target_partition (str): The target partition to fix.
		fs_type (str, optional): The file system type of the target partition. If not provided, it will be determined automatically.

	Returns:
		bool: True if the file system errors are fixed successfully, False otherwise.
	"""
	try:
		if not fs_type:
			fs_type = get_fs_type(target_partition)
		# Fix it!
		if fs_type == 'ext4' or fs_type == 'ext3' or fs_type == 'ext2':
			#run_command_in_multicmd_with_path_check(f"e2fsck -f -y {target_partition}",strict=False)
			run_command_in_multicmd_with_path_check(["e2fsck", '-f', '-y', target_partition],strict=False)
		elif fs_type == 'btrfs':
			#run_command_in_multicmd_with_path_check(f"btrfs check --repair {target_partition}",strict=False)
			run_command_in_multicmd_with_path_check(["btrfs", 'check', '--repair', target_partition],strict=False)
		elif fs_type == 'xfs':
			#run_command_in_multicmd_with_path_check(f"xfs_repair -L {target_partition}",strict=False)
			run_command_in_multicmd_with_path_check(["xfs_repair", '-L', target_partition],strict=False)
		elif fs_type == 'ntfs':
			#run_command_in_multicmd_with_path_check(f"ntfsfix {target_partition}",strict=False)
			run_command_in_multicmd_with_path_check(["ntfsfix", target_partition],strict=False)
		elif fs_type == 'fat32' or fs_type == 'fat16' or fs_type == 'fat12' or fs_type == 'fat' or fs_type == 'vfat':
			#run_command_in_multicmd_with_path_check(f"fsck.fat -w -r -l -a -v {target_partition}",strict=False)
			run_command_in_multicmd_with_path_check(["fsck.fat", '-w', '-r', '-l', '-a', '-v', target_partition],strict=False)
		elif fs_type == 'exfat':
			#run_command_in_multicmd_with_path_check(f"fsck.exfat -p -y {target_partition}",strict=False)
			run_command_in_multicmd_with_path_check(["fsck.exfat", '-p', '-y', target_partition],strict=False)
		elif fs_type == 'hfsplus':
			#run_command_in_multicmd_with_path_check(f"fsck.hfsplus -y {target_partition}",strict=False)
			run_command_in_multicmd_with_path_check(["fsck.hfsplus", '-y', target_partition],strict=False)
		elif fs_type == 'hfs':
			#run_command_in_multicmd_with_path_check(f"fsck.hfs -y {target_partition}",strict=False)
			run_command_in_multicmd_with_path_check(["fsck.hfs", '-y', target_partition],strict=False)
		elif fs_type == 'jfs':
			#run_command_in_multicmd_with_path_check(f"fsck.jfs -y {target_partition}",strict=False)
			run_command_in_multicmd_with_path_check(["fsck.jfs", '-y', target_partition],strict=False)
		elif fs_type == 'reiserfs':
			#run_command_in_multicmd_with_path_check(f"fsck.reiserfs -y {target_partition}",strict=False)
			run_command_in_multicmd_with_path_check(["fsck.reiserfs", '-y', target_partition],strict=False)
		elif fs_type == 'udf':
			print(f"Warning: Cannot fix udf file system. Skipping.")
		elif fs_type == 'ufs':
			#run_command_in_multicmd_with_path_check(f"fsck.ufs -y {target_partition}",strict=False)
			run_command_in_multicmd_with_path_check(["fsck.ufs", '-y', target_partition],strict=False)
		elif fs_type == 'bfs':
			print(f"Warning: Cannot fix bfs file system. Skipping.")
		elif fs_type == 'minix':
			#run_command_in_multicmd_with_path_check(f"fsck.minix -a {target_partition}",strict=False)
			run_command_in_multicmd_with_path_check(["fsck.minix", '-a', target_partition],strict=False)
		else:
			print(f"File system {fs_type} not supported.")
			return False
	except Exception as e:
		print(f"Exception caught when trying to fix {target_partition} with {fs_type}: {e}")
		return False
	return True

def create_loop_device(image_path,read_only=False):
	"""Create a loop device for the image file."""
	ro = '--read-only' if read_only else ''
	loop_device_dest = run_command_in_multicmd_with_path_check(["losetup", '--partscan', '--find', '--show', ro, image_path])[0].strip()
	#run_command_in_multicmd_with_path_check(f'partprobe {loop_device_dest}')
	print(f"Loop device {loop_device_dest} created.")
	return loop_device_dest

def get_target_partition(image, partition_name):
	loop_device = None
	if not pathlib.Path(image).resolve().is_block_device():
		loop_device = create_loop_device(image)
		image = loop_device
	# Need to get a partition path for mkfs
	partitions = run_command_in_multicmd_with_path_check(["lsblk", '-nbl', '-o', 'NAME', image])
	partitions.pop(0) # remove the disk itself
	target_partition = ''
	for part in partitions:
		if part.endswith(partition_name):
			target_partition = '/dev/' + part
			break
	return target_partition, loop_device

@functools.lru_cache(maxsize=None)
def get_partition_details(device, partition,sector_size=512):
	"""Get the partition details of the partition."""
	# Get the partition info from the source
	result = run_command_in_multicmd_with_path_check(["sgdisk", '--info='+partition, device])
	rtnDic = {'partition_guid_code': '', 'unique_partition_guid': '', 'partition_name': '', 'partition_attrs': '', 'fs_type': '', 'fs_uuid': '', 'fs_label': '', 'size': 0}
	for line in result:
		if "guid code:" in line.lower():
			rtnDic['partition_guid_code'] = line.split(":")[1].split()[0].strip()
		elif "unique guid:" in line.lower():
			rtnDic['unique_partition_guid'] = line.split(":")[1].strip()
		elif "partition name:" in line.lower():
			rtnDic['partition_name'] = line.split("'")[1].strip()
		elif "attribute flags:" in line.lower():
			rtnDic['partition_attrs'] = line.split(":")[1].strip()
		elif 'partition size:' in line.lower():
			rtnDic['size'] = int(line.split(':')[1].split()[0].strip()) * int(sector_size)
	# Also get the fs type, use parted
	# result = run_command_in_multicmd_with_path_check(f"parted --machine --script {device} print")
	# for line in result:
	# 	if line.startswith(partition):
	# 		rtnDic['fs_type'] = line.split(':')[4].strip()
	# 		return rtnDic
	# Get the fs type and uuid, use blkid
	target_partition, loop_device = get_target_partition(device, partition)
	result = run_command_in_multicmd_with_path_check(["blkid", '-o', 'export', target_partition])
	for line in result:
		if 'TYPE' in line.upper():
			rtnDic['fs_type'] = line.split('=')[1].strip()
		elif 'UUID' in line.upper() and 'PARTUUID' not in line.upper():
			rtnDic['fs_uuid'] = line.split('=')[1].strip()
		elif 'LABEL' in line.upper() and 'PARTLABEL' not in line.upper():
			rtnDic['fs_label'] = line.split('=')[1].strip()
	if loop_device:
		run_command_in_multicmd_with_path_check(["losetup", '--detach', loop_device])
	return rtnDic

@functools.lru_cache(maxsize=None)
def get_partition_infos(device):
	"""Get partition information of the device."""
	# partitions = run_command_in_multicmd_with_path_check(f"lsblk -nbl -o NAME,SIZE {device}")
	# partition_info = {part.split()[0]: int(part.split()[1]) for part in partitions}
	partitions = run_command_in_multicmd_with_path_check(["sgdisk", '--print', device])
	line = partitions.pop(0)
	partition_info = {}
	disk_size_sector = 0
	disk_identifier = ''
	sector_size = 512
	disk_name = ''
	while not line.lower().startswith('number'):
		if line.lower().startswith('disk') and not line.lower().startswith('disk identifier') and 'sectors' in line:
			disk_size_sector = int(line.rpartition(':')[2].split()[0])
			disk_name = line.split()[1].strip(':')
		elif 'sector size' in line.lower():
			sector_size = int(line.rpartition(':')[2].split()[0].partition('/')[0])
		elif 'disk identifier' in line.lower():
			disk_identifier = line.rpartition(':')[2].strip()
		line = partitions.pop(0)
	partition_info[disk_name] = {'size': disk_size_sector*sector_size, 'disk_identifier': disk_identifier, 'sector_size': sector_size}
	for part in partitions:
		if not part or not part.strip() or len(part.split()) < 4 or part.lower().startswith('error'):
			continue
		part = part.split()
		partition_info[part[0]] = get_partition_details(device, part[0],sector_size)
	return partition_info

def write_partition_info(image, partition_infos, partition_name):
	"""
	Copies partition information from one partition to another using sgdisk.

	:param image: The image file to write the partition information to.
	:param partition_info: The partition information.
	"""
	try:
		# Apply the GUID code, unique GUID, and attributes to the target
		if partition_infos[partition_name]['partition_guid_code']:
			run_command_in_multicmd_with_path_check(["sgdisk", '--typecode='+partition_name+':'+partition_infos[partition_name]['partition_guid_code'], image],strict=False)
		if partition_infos[partition_name]['unique_partition_guid']:
			run_command_in_multicmd_with_path_check(["sgdisk", '--partition-guid='+partition_name+':'+partition_infos[partition_name]['unique_partition_guid'], image],strict=False)
		if partition_infos[partition_name]['partition_attrs']:
			binary_attributes = bin(int(partition_infos[partition_name]['partition_attrs'], 16))[2:].zfill(64)[::-1]
			for i, bit in enumerate(binary_attributes):
				if bit == '1':
					run_command_in_multicmd_with_path_check(["sgdisk", f"--attributes={partition_name}:set:{i}", image],strict=False)
		if partition_infos[partition_name]['fs_type']:
			target_partition, loop_device = get_target_partition(image, partition_name)
			if not target_partition:
				print(f"Error: Cannot find partition {partition_name} in {image}.")
				return
			fs_type = partition_infos[partition_name]['fs_type']
			fs_label = partition_infos[partition_name]['fs_label']
			fs_uuid = partition_infos[partition_name]['fs_uuid']
			if fs_type == 'ext4' or fs_type == 'ext3' or fs_type == 'ext2':
				command = ['mkfs', '-t', fs_type]
				if fs_label:
					command.extend(['-L', fs_label])
				if fs_uuid:
					command.extend(['-U', fs_uuid])
				command.append(target_partition)
				run_command_in_multicmd_with_path_check(command,strict=False)
			elif fs_type == 'btrfs':
				commad = ['mkfs.btrfs']
				if fs_label:
					command.extend(['-L', fs_label])
				if fs_uuid:
					command.extend(['-U', fs_uuid])
				command.append(target_partition)
				run_command_in_multicmd_with_path_check(command,strict=False)
			elif fs_type == 'xfs':
				command = ['mkfs.xfs']
				if fs_label:
					command.extend(['-L', fs_label])
				if fs_uuid:
					command.extend(['-m', f'uuid={fs_uuid}'])
				command.append(target_partition)
				run_command_in_multicmd_with_path_check(command,strict=False)
			elif fs_type == 'ntfs':
				command = ['mkfs.ntfs']
				if fs_label:
					command.extend(['-L', fs_label])
				command.append(target_partition)
				run_command_in_multicmd_with_path_check(command,strict=False)
				if fs_uuid:
					print(f"Warning: Cannot set fs uuid for ntfs. Skipping.")
			elif fs_type == 'fat32' or fs_type == 'fat16' or fs_type == 'fat12' or fs_type == 'fat' or fs_type == 'vfat' or fs_type == 'msdos':
				command = ['mkfs.vfat']
				if fs_type == 'fat16':
					command.extend(['-F', '16'])
				elif fs_type == 'fat12':
					command.extend(['-F', '12'])
				if fs_label:
					command.extend(['-n', fs_label])
				if fs_uuid:
					command.extend(['-i', fs_uuid.lower().replace('-','')])
				command.append(target_partition)
				run_command_in_multicmd_with_path_check(command,strict=False)
			elif fs_type == 'exfat':
				command = ['mkfs.exfat']
				if fs_label:
					command.extend(['-L', fs_label])
				command.append(target_partition)
				run_command_in_multicmd_with_path_check(command,strict=False)
				if fs_uuid:
					run_command_in_multicmd_with_path_check(["exfatlabel", '-i', target_partition, fs_uuid],strict=False)
			elif fs_type == 'hfsplus' or fs_type == 'hfs':
				command = [f'mkfs.{fs_type}']
				if fs_label:
					command.extend(['-v', fs_label])
				command.append(target_partition)
				run_command_in_multicmd_with_path_check(command,strict=False)
				if fs_uuid:
					print(f"Warning: Cannot set fs uuid for {fs_type}. Skipping.")
			elif fs_type == 'udf':
				# command += f" {target_partition}"
				command = ['mkudffs', '--media-type=hd']
				if fs_label:
					command.extend(['--label', fs_label])
				if fs_uuid:
					command.extend(['--uuid', fs_uuid])
				command.append(target_partition)
				run_command_in_multicmd_with_path_check(command,strict=False)
			elif fs_type == 'jfs':
				command = ['mkfs.jfs']
				if fs_label:
					command.extend(['-L', fs_label])
				command.append(target_partition)
				run_command_in_multicmd_with_path_check(command,strict=False)
				if fs_uuid:
					run_command_in_multicmd_with_path_check(["jfs_tune", '-U', fs_uuid, target_partition],strict=False)
			elif fs_type == 'reiserfs':
				command = ['mkfs.reiserfs']
				if fs_label:
					command.extend(['-l', fs_label])
				if fs_uuid:
					command.extend(['-u', fs_uuid])
				command.append(target_partition)
				run_command_in_multicmd_with_path_check(command,strict=False)
			elif fs_type == 'zfs':
				print(f"Skip creating zfs file system. ZFS file system should be created using zpool command.")
			elif fs_type == 'ufs':
				command = ['newfs', '-t']
				if fs_label:
					command.extend(['-L', fs_label])
				command.append(target_partition)
				run_command_in_multicmd_with_path_check(command,strict=False)
				if fs_uuid:
					print(f"Warning: Cannot set fs uuid for ufs. Skipping.")
			elif fs_type == 'bfs':
				command = ['mkfs.bfs']
				if fs_label:
					command.extend(['-F', fs_label, '-V', fs_label])
				command.append(target_partition)
				run_command_in_multicmd_with_path_check(command,strict=False)
				if fs_uuid:
					print(f"Warning: Cannot set fs uuid for bfs. Skipping.")
			elif fs_type == 'cramfs':
				print(f"Warning: cramfs is read-only file system. You should create one with mkfs.cramfs.")
			elif fs_type == 'minix':
				command = ['mkfs.minix',target_partition]
				run_command_in_multicmd_with_path_check(command,strict=False)
				if fs_label:
					print(f"Warning: Cannot set fs label for minix. Skipping.")
				if fs_uuid:
					print(f"Warning: Cannot set fs uuid for minix. Skipping.")
			elif fs_type == 'iso9660':
				print(f"Warning: iso9660 is read-only file system. You should create one with mkfs.iso9660.")
			elif fs_type == 'swap':
				command = ['mkswap']
				if fs_label:
					command.extend(['-L', fs_label])
				if fs_uuid:
					command.extend(['-U', fs_uuid])
				command.append(target_partition)
				run_command_in_multicmd_with_path_check(command,strict=False)
			elif fs_type == 'gpt':
				print(f"Skip creating gpt padding.")
			else:
				print(f"Warning: File system {fs_type} not currently supported by hpcp. Trying mkfs -t {fs_type} anyway...")
				command = ['mkfs', '-t', fs_type]
				if fs_label:
					command.extend(['-L', fs_label])
				if fs_uuid:
					command.extend(['-U', fs_uuid])
				command.append(target_partition)
				run_command_in_multicmd_with_path_check(command,strict=False)
			if loop_device:
				run_command_in_multicmd_with_path_check(["losetup", '--detach', loop_device])
		if partition_infos[partition_name]['partition_name']:
			run_command_in_multicmd_with_path_check(['sgdisk', f'--change-name={partition_name}:{partition_infos[partition_name]["partition_name"]}', image],strict=False)


	except Exception as e:
		print("An error occurred while copying partition information:", e)

def create_partition_table(image, partition_infos,sorted_partitions):
	"""Create a partition table in the image file that will match the source device and sort the partitions by size."""
	# Create a partition table in the image file
	run_command_in_multicmd_with_path_check(["sgdisk", '--clear', image])
	src_disk_path = sorted_partitions.pop()
	# set the disk identifier ( disk UUID )
	run_command_in_multicmd_with_path_check(["sgdisk", f'--disk-guid={partition_infos[src_disk_path]["disk_identifier"]}', image])
	# Create the filesystem, need to mount the image first if it is a image
	loop_device = None
	if not pathlib.Path(image).resolve().is_block_device():
		loop_device = create_loop_device(image)
		image = loop_device
	# Create the partitions
	for partition in sorted_partitions:
		start_sector = int(run_command_in_multicmd_with_path_check(["sgdisk", '--first-aligned-in-largest', image])[0].strip())
		end_sector = start_sector + int(partition_infos[partition]['size']/partition_infos[src_disk_path]['sector_size']) -1
		# Create the partition
		run_command_in_multicmd_with_path_check(["sgdisk", f'--new={partition}:{start_sector}:{end_sector}', image])
		# Copy the partition information
		write_partition_info(image, partition_infos,partition)
	# Fix the partition table
	run_command_in_multicmd_with_path_check(["sgdisk", '--verify', image])
	if loop_device:
		run_command_in_multicmd_with_path_check(["losetup", '--detach', loop_device])

def resize_image(image, total_size):
	"""Resize the image file to the calculated size."""
	# use truncate to create a file if the image is not a block device
	if not pathlib.Path(image).resolve().is_block_device():
		run_command_in_multicmd_with_path_check(["truncate", '--size='+format_bytes(total_size,to_int=True), image])
def creatSymLinks(symLinks,exclude=None,no_link_tracking=False):
	if len(symLinks) == 0:
		return
	nestedSymLinks = {}
	counter = 0
	print(f"\nFound Symbolic Links:   {len(symLinks)}")
	if no_link_tracking:
		print(f"Skipping copying file as no_link_tracking ...\n")
	#print(symLinks)
	startTime = time.perf_counter()
	for src, dest in symLinks.items():
		try:
			src = os.path.normpath(src)
			dest = os.path.normpath(dest)
			if exclude and is_excluded(src,exclude):
				print(f"\n{src} is excluded, skipping...")
				continue
			if os.path.islink(dest):
				os.unlink(dest)
			if os.path.exists(dest):
				if os.path.isdir(dest):
					print(f"\n{dest} is a directory, skipping...")
					#shutil.rmtree(dest)
					continue
				else:
					print(f"\n{dest} is a file, skipping...")
					#os.remove(dest)
					continue

			# Determine if the link is a absolute link or relative link
			linkedTargetFile = os.readlink(src)
			if not os.path.isabs(linkedTargetFile):
				sourceLinkedFile = os.path.join(os.path.dirname(src), linkedTargetFile)
				# we also copy the pointed file if the file doesn't exist
				destLinkedFile = os.path.join(os.path.dirname(dest), linkedTargetFile)
				#print(f"sourcelinkedfile: {sourceLinkedFile} -> destlinkedfile: {destLinkedFile}")
				if not os.path.exists(destLinkedFile):
					if no_link_tracking:
						pass
					elif not os.path.exists(sourceLinkedFile):
						print(f"\nFile {sourceLinkedFile} which is linked by {src} doesn't exist! \nSkipping copying original file...")
					else:
						# if os.path.islink(sourceLinkedFile):
						#     nestedSymLinks[sourceLinkedFile] = destLinkedFile
						# elif os.path.isdir(sourceLinkedFile):
						#     #shutil.copytree(sourceLinkedFile, destLinkedFile, symlinks=True, ignore_dangling_symlinks=True)
						#     # we use copy_file_serial because python copytree sucks
						#     _, _ , rtnSymLinks , _ = copy_files_serial(sourceLinkedFile, destLinkedFile)
						#     nestedSymLinks.update(rtnSymLinks)
						# elif os.path.isfile(sourceLinkedFile):
						#     # need to create directories if they don't exist
						#     os.makedirs(os.path.dirname(destLinkedFile), exist_ok=True)
						#     _, _ , rtnSymLinks , _ = copy_files_serial(sourceLinkedFile, destLinkedFile)
						#     nestedSymLinks.update(rtnSymLinks)
						# else:
						#     print(f"\nFile {sourceLinkedFile} which is linked by {src} is not a file, directory or symbolic link!")
						#     if os.name == 'posix':
						#         os.system(f"cp -af {sourceLinkedFile} {destLinkedFile}")
						#     elif os.name == 'nt':
						#         os.system(f"xcopy /I /E /Y /c /q /k /r /h /x {sourceLinkedFile} {destLinkedFile}")
						_, _ , rtnSymLinks , _ = copy_files_serial(sourceLinkedFile, destLinkedFile,exclude=exclude)
						nestedSymLinks.update(rtnSymLinks)
			try:
				os.symlink(linkedTargetFile, dest, target_is_directory=os.path.isdir(linkedTargetFile))
			except:
				print(f'Could not create symbolic link from {linkedTargetFile} to {dest}')
			counter += 1


			# print the progress bar with the total count and the speed in F/s
			prefix = f'{counter} Symbolic Links Created'
			suffix = f'{counter / (time.perf_counter() - startTime):.2f} F/s'
			multiCMD.print_progress_bar(counter, len(symLinks), prefix=prefix, suffix=suffix)
			# we catch the file name too long exception
		except OSError as e:
			print("Exception caught! Possibly file name too long!")
			print(f"\n{e}")
			print(f"\n{src} -> {dest}")
			print("Skipping...")
			continue
		except Exception as e:
			print("Exception caught!")
			print(f"\n{e}")
			print(f"\n{src} -> {dest}")
			print("Skipping...")
			continue


	endTime = time.perf_counter()
	print(f"Time taken:             {endTime-startTime:0.4f} seconds")
	if len(nestedSymLinks) > 0:
		print(f"\nNested Symbolic Links:   {len(nestedSymLinks)}")
		creatSymLinks(nestedSymLinks,exclude=exclude,no_link_tracking=no_link_tracking)

# -- File list --
def natural_sort(l): 
	"""
	Sorts a list of strings naturally, considering both numeric and alphabetic characters.

	Args:
		l (list): The list of strings to be sorted.

	Returns:
		list: The sorted list of strings.
	"""
	convert = lambda text: int(text) if text.isdigit() else text.lower() 
	alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)] 
	return sorted(l, key=alphanum_key)

# -- Format --
def format_bytes(size, use_1024_bytes=None, to_int=False, to_str=False,str_format='.2f'):
	"""
	Format the size in bytes to a human-readable format or vice versa.

	Args:
		size (int or str): The size in bytes or a string representation of the size.
		use_1024_bytes (bool, optional): Whether to use 1024 bytes as the base for conversion. If None, it will be determined automatically. Default is None.
		to_int (bool, optional): Whether to convert the size to an integer. Default is False.
		to_str (bool, optional): Whether to convert the size to a string representation. Default is False.
		str_format (str, optional): The format string to use when converting the size to a string. Default is '.2f'.

	Returns:
		int or str: The formatted size based on the provided arguments.

	Examples:
		>>> format_bytes(1500)
		'1.50 KB'
		>>> format_bytes('1.5 GiB', to_int=True)
		1610612736
	"""
	if to_int or isinstance(size, str):
		if isinstance(size, int):
			return size
		elif isinstance(size, str):
			# Use regular expression to split the numeric part from the unit, handling optional whitespace
			match = re.match(r"(\d+(\.\d+)?)\s*([a-zA-Z]*)", size)
			if not match:
				print("Invalid size format. Expected format: 'number [unit]', e.g., '1.5 GiB' or '1.5GiB'")
				print(f"Got: {size}")
				return 0
			number, _, unit = match.groups()
			number = float(number)
			unit  = unit.strip().lower().rstrip('b')
			# Define the unit conversion dictionary
			if unit.endswith('i'):
				# this means we treat the unit as 1024 bytes if it ends with 'i'
				use_1024_bytes = True
			elif use_1024_bytes is None:
				use_1024_bytes = False
			unit  = unit.rstrip('i')
			if use_1024_bytes:
				power = 2**10
			else:
				power = 10**3
			unit_labels = {'': 0, 'k': 1, 'm': 2, 'g': 3, 't': 4, 'p': 5}
			if unit not in unit_labels:
				print(f"Invalid unit '{unit}'. Expected one of {list(unit_labels.keys())}")
				return 0
			# Calculate the bytes
			return int(number * (power ** unit_labels[unit]))
		else:
			try:
				return int(size)
			except Exception as e:
				return 0
	elif to_str or isinstance(size, int) or isinstance(size, float):
		if isinstance(size, str):
			try:
				size = size.lower().strip().rstrip('b')
				size = float(size)
			except Exception as e:
				return size
		# size is in bytes
		if use_1024_bytes or use_1024_bytes is None:
			power = 2**10
			n = 0
			power_labels = {0 : '', 1: 'Ki', 2: 'Mi', 3: 'Gi', 4: 'Ti', 5: 'Pi'}
			while size > power:
				size /= power
				n += 1
			return f"{size:{str_format}} {power_labels[n]}"
		else:
			power = 10**3
			n = 0
			power_labels = {0 : '', 1: 'K', 2: 'M', 3: 'G', 4: 'T', 5: 'P'}
			while size > power:
				size /= power
				n += 1
			return f"{size:{str_format}} {power_labels[n]}"
	else:
		try:
			return format_bytes(float(size), use_1024_bytes)
		except Exception as e:
			import traceback
			print(f"Error: {e}")
			print(traceback.format_exc())
			print(f"Invalid size: {size}")
		return 0

def format_time(seconds):
	"""
	Format the time in seconds to a short, human-readable format.

	Args:
		seconds (int): The time in seconds.

	Returns:
		str: The formatted time in a human-readable format.
	"""
	try:
		seconds = int(seconds)
	except:
		return seconds
	seconds_in_minute = 60
	seconds_in_hour = 60 * seconds_in_minute
	seconds_in_day = 24 * seconds_in_hour
	seconds_in_month = 30 * seconds_in_day
	seconds_in_year = 365 * seconds_in_day
	years, seconds = divmod(seconds, seconds_in_year)
	months, seconds = divmod(seconds, seconds_in_month)
	days, seconds = divmod(seconds, seconds_in_day)
	hours, seconds = divmod(seconds, seconds_in_hour)
	minutes, seconds = divmod(seconds, seconds_in_minute)
	parts = [
		f"{years:.0f}Y" if years else "",
		f"{months:.0f}M" if months else "",
		f"{days:.0f}D" if days else "",
		f"{hours:.0f}h" if hours else "",
		f"{minutes:.0f}m" if minutes else "",
		f"{seconds:.0f}s" if seconds else "0s",
	]
	return "".join(parts)

# -- Hash --
@functools.lru_cache(maxsize=None)
def hash_file(path,size = ...,full_hash=False):
	global xxhash_available
	global HASH_SIZE
	if HASH_SIZE <= 0:
		# Do not hash
		return ''
	if size == ...:
		size = os.path.getsize(path)
	hasher = xxhash.xxh64() if xxhash_available else hashlib.blake2b()
	with open(path, 'rb') as f:
		if not full_hash:
			# Only hash the last hash_size bytes
			#f.seek(-min(1<<16,size), os.SEEK_END)
			f.seek(-min(HASH_SIZE,size), os.SEEK_END)
		for chunk in iter(lambda: f.read(4096), b''):
			hasher.update(chunk)
	return hasher.hexdigest()

def is_file_identical(src_path, dest_path,src_size,full_hash=False):
	dst_size = os.path.getsize(dest_path)
	# try to find the mtime are different
	try:
		src_mtime = os.path.getmtime(src_path)
		dst_mtime = os.path.getmtime(dest_path)
		if src_mtime != dst_mtime:
			return False
	except:
		pass
	return src_size == dst_size and hash_file(src_path,src_size,full_hash) == hash_file(dest_path,dst_size,full_hash)

def get_file_repr(filename,append_hash=False,full_hash=False):
	if append_hash:
		if os.path.islink(filename):
			return f'{filename}:{os.readlink(filename)}'
		return f'{filename}:{hash_file(filename,os.path.getsize(filename),full_hash)}'
	return filename

# -- Path --
def trimPaths(paths,baseDir):
	return set([os.path.relpath(path,os.path.dirname(baseDir)) for path in paths])
# ---- Generate File List ----
@functools.lru_cache(maxsize=None)
def get_file_list_serial(root,exclude=None,append_hash=False,full_hash=False):
	# skip if path is longer than 4096 characters
	if len(root) > 4096:
		return frozenset() ,frozenset(),0,frozenset()
	#print(f'Getting file list for {root}')
	if exclude and is_excluded(root,exclude):
		return frozenset() ,frozenset(),0,frozenset()
	if os.path.islink(root):
		return frozenset() ,frozenset([get_file_repr(root,append_hash,full_hash)]),0,frozenset()
	if os.path.isfile(root):
		st = os.stat(root, follow_symlinks=False)
		realSize = st.st_rsize if 'st_rsize' in st else st.st_blocks * 512 if 'st_blocks' in st else st.st_size
		return frozenset([get_file_repr(root,append_hash,full_hash)]) ,frozenset(), realSize,frozenset()
	file_list = set()
	links = set()
	folders = set()
	size = 0
	iteration = 0
	startTime = time.perf_counter()
	globalStartTIme = startTime
	if os.path.isdir(root):
		folders.add(root)
		for entry in os.scandir(root):
			# update the progress bar every 0.5 seconds
			currentTime = time.perf_counter()
			if currentTime - startTime > 0.5:
				startTime = currentTime
				# use the time passed as the iteration number
				iteration = int(currentTime - globalStartTIme)
				# if the root is longer than 50 characters, we only show the last 50 characters
				multiCMD.print_progress_bar(iteration=iteration, total=0, prefix=f'{root}'[-50:], suffix=f'Files: {format_bytes(len(file_list),use_1024_bytes=False,to_str=True)} Links: {format_bytes(len(links),use_1024_bytes=False,to_str=True)} Folders: {format_bytes(len(folders),use_1024_bytes=False,to_str=True)} Size: {format_bytes(size)}B')
			if exclude and is_excluded(entry.path,exclude):
				continue
			if entry.is_symlink():
				links.add(get_file_repr(entry.path,append_hash,full_hash))
				try:
					st = entry.stat(follow_symlinks=False)
					realSize = st.st_rsize if 'st_rsize' in st else st.st_blocks * 512 if 'st_blocks' in st else st.st_size
				except:
					realSize = os.path.getsize(entry.path)
				size += realSize
			elif entry.is_file(follow_symlinks=False):
				file_list.add(get_file_repr(entry.path,append_hash,full_hash))
				try:
					st = entry.stat(follow_symlinks=False)
					realSize = st.st_rsize if 'st_rsize' in st else st.st_blocks * 512 if 'st_blocks' in st else st.st_size
				except:
					try:
						realSize = os.path.getsize(entry.path)
					except:
						realSize = 0
				size += realSize
			elif entry.is_dir(follow_symlinks=False):
				dir_files, dir_links, dir_size, dir_folders = get_file_list_serial(entry.path,exclude=exclude,append_hash=append_hash,full_hash=full_hash)
				file_list.update(dir_files)
				links.update(dir_links)
				size += dir_size
				folders.update(dir_folders)
		#multiCMD.print_progress_bar(iteration=iteration, total=iteration, prefix=f'{root}', suffix=f'Files: {format_bytes(len(file_list),use_1024_bytes=False,to_str=True)} Links: {format_bytes(len(links),use_1024_bytes=False,to_str=True)} Folders: {format_bytes(len(folders),use_1024_bytes=False,to_str=True)} Size: {format_bytes(size)}B')
	else:
		print(f'Error: {root} is not a file or directory')
		return frozenset([root]) ,frozenset(), 0,frozenset()
	return frozenset(file_list), frozenset(links) , size, frozenset(folders - set(['.', '..']))

@functools.lru_cache(maxsize=None)
def get_file_list_parallel(path,max_workers=56,exclude=None,append_hash=False,full_hash=False):
	# skip if path is longer than 4096 characters
	if len(path) > 4096:
		return frozenset() ,frozenset(),0,frozenset()
	if exclude and is_excluded(path,exclude):
		return frozenset() ,frozenset(),0,frozenset()
	if os.path.islink(path):
		return frozenset() ,frozenset([get_file_repr(path,append_hash,full_hash)]),0,frozenset()
	if os.path.isfile(path):
		st = os.stat(path, follow_symlinks=False)
		realSize = st.st_rsize if 'st_rsize' in st else st.st_blocks * 512 if 'st_blocks' in st else st.st_size
		return frozenset([get_file_repr(path,append_hash,full_hash)]) ,frozenset(), realSize,frozenset()

	
	print(f'Getting file list for {path}')
	
	if os.path.isdir(path):
		file_list = set()
		link_list = set()
		size = 0
		folder_list = set()
		with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
			futures = []
			for subPath in os.listdir(path):
				fullPath = os.path.join(path, subPath)
				if exclude and is_excluded(fullPath,exclude):
					continue
				if os.path.islink(fullPath):
					link_list.add(get_file_repr(fullPath,append_hash,full_hash))
				elif os.path.isfile(fullPath):
					file_list.add(get_file_repr(fullPath,append_hash,full_hash))
				elif os.path.isdir(fullPath):
					futures.append(executor.submit(get_file_list_serial, fullPath,exclude=exclude,append_hash=append_hash,full_hash=full_hash))
					folder_list.add(fullPath)
				else:
					print(f'Unknown file type: {fullPath}')
					file_list.add(fullPath)
			
			for future in concurrent.futures.as_completed(futures):
				files, links, sizes, folders = future.result()
				file_list.update(files)
				link_list.update(links)
				size += sizes
				folder_list.update(folders)
	else:
		print(f'Unknown file type: {path}')
		return frozenset([path]) ,frozenset(), 0,frozenset()
	return frozenset(file_list), frozenset(link_list) ,size, frozenset(folder_list - set(['.', '..']))

# ---- Delete Files ----
def delete_file_bulk(paths):
	total_size = 0
	startTime = time.perf_counter()
	for path in paths:
		if os.path.exists(path):
			try:
				total_size += os.path.getsize(path)
				if os.path.islink(path):
					os.unlink(path)
				elif os.path.isdir(path):
					shutil.rmtree(path)
				else:
					os.remove(path)
			except Exception as exc:
				print(f'\nDeleting {path} generated an exception: {exc}')
	endTime = time.perf_counter()
	return total_size, endTime - startTime

def delete_file_list_parallel(file_list, max_workers, verbose=False,files_per_job=1,init_size=0):
	total_files = len(file_list)
	file_list_iterator = iter(file_list)
	startTime = time.perf_counter()
	last_refresh_time = startTime
	futures = {}
	files_per_job = max(1,files_per_job)
	apb = Adaptive_Progress_Bar(total_count=total_files,total_size=init_size,last_num_job_for_stats=max(1,max_workers // 2),process_word='Deleted',use_print_thread = True)
	with ProcessPoolExecutor(max_workers=max_workers) as executor:
		while file_list_iterator or futures:
			# counter = 0
			while file_list_iterator and len(futures) < 1.2 * max_workers and last_refresh_time - time.perf_counter() < 5:
				delete_files = []
				try:
					for _ in range(files_per_job):
						delete_files.append(next(file_list_iterator))
					future = executor.submit(delete_file_bulk, delete_files)
					futures[future] = delete_files
					# counter += 1
				except StopIteration:
					if delete_files:
						future = executor.submit(delete_file_bulk, delete_files)
						futures[future] = delete_files
					file_list_iterator = None

			done, _ = concurrent.futures.wait(futures.keys(), return_when=concurrent.futures.FIRST_COMPLETED)

			#print('\n',len(done) ,'\t', len(futures))
			# if file_list_iterator and len(done) / len(futures) > 0.2:
			# 	if verbose:
			# 		print(f'\nTransfer is fast, doubling files per job to {files_per_job * 2}')
			# 	files_per_job *= 2

			current_iteration_total_run_time = 0
			for future in done:
				delete_files = futures.pop(future)
				deleted_files_count_this_run = len(delete_files)
				rmSize = rmTime = 0
				try:
					rmSize, rmTime = future.result()
				except Exception as exc:
					print(f'\n{future} generated an exception: {exc}')
				current_iteration_total_run_time += rmTime
				apb.update(num_files=deleted_files_count_this_run,cpSize=rmSize,cpTime=rmTime,files_per_job=files_per_job)
				apb.scheduled_jobs = len(futures)
			if verbose:
					print(f'\nAverage rmtime is {current_iteration_total_run_time / len(done):0.2f} for {len(done)} jobs with {deleted_files_count_this_run} files each')

			if file_list_iterator and deleted_files_count_this_run == files_per_job and (current_iteration_total_run_time / len(done) > 5 or time.perf_counter() - last_refresh_time > 5):
				files_per_job //= 1.61803398875
				files_per_job = round(files_per_job)
				if verbose:
					print(f'\nCompletion time is long, changing files per job to {files_per_job}')
			elif file_list_iterator and deleted_files_count_this_run == files_per_job and current_iteration_total_run_time / len(done) < 1:
				files_per_job *= 1.61803398875
				files_per_job = round(files_per_job)
				if verbose:
					print(f'\nCompletion time is short, changing files per job to {files_per_job}')
			if files_per_job < 1:
				files_per_job = 1
			last_refresh_time = time.perf_counter()
	endTime = time.perf_counter()
	apb.stop()
	print(f"\nTime taken:             {endTime-startTime:0.4f} seconds")
	print(f"Average speed:          {format_bytes(apb.size_counter / (endTime-startTime))}B/s")
	print(f"                        {apb.item_counter / (endTime-startTime):.2f} file/s")
	return apb.item_counter, apb.size_counter

def delete_files_parallel(paths, max_workers, verbose=False,files_per_job=1,exclude=None,batch=False):
	if not batch and isinstance(paths, str):
		paths = [paths]
	startTime = time.perf_counter()
	all_files = set()
	init_size_all = 0
	for path in paths:
		file_list, links,init_size, folders = get_file_list_serial(path,exclude=exclude)
		#file_list, links,init_size, folders = get_file_list_parallel(path, max_workers,exclude=exclude)
		all_files.update(set(file_list) | set(links))
		init_size_all += init_size
	endTime = time.perf_counter()
	print(f"Time taken to get file list: {endTime-startTime:0.4f} seconds")
	total_files = len(all_files)
	print(f"Number of files: {total_files}")
	if total_files == 0:
		return 1 , delete_file_bulk(paths)[0]
	delete_counter, delete_size_counter = delete_file_list_parallel(all_files, max_workers, verbose,files_per_job,init_size=init_size)
	print("Removing directory structures....")
	delete_size_counter += delete_file_bulk(paths)[0]
	print(f"Initial estimated size: {format_bytes(init_size_all)}B, Final size: {format_bytes(delete_size_counter)}B")
	return delete_counter + 1, delete_size_counter

# ---- Copy Files ----
def copy_file(src_path, dest_path, full_hash=False, verbose=False):
	"""
	Copy a file from the source path to the destination path.

	Args:
		src_path (str): The path of the source file.
		dest_path (str): The path of the destination file.
		full_hash (bool, optional): Whether to perform a full hash comparison to determine if the files are identical. Defaults to False.
		verbose (bool, optional): Whether to print verbose output. Defaults to False.

	Returns:
		tuple: A tuple containing the size of the copied file, the time taken for the copy operation, and a dictionary of symbolic links encountered during the copy.
	"""
	symLinks = {}
	#task_to_run = []
	# skip if src path or dest path is longer than 4096 characters
	if len(src_path) > 4096:
		print(f'\nSkipped {src_path} because path is too long')
		return 0, 0, symLinks #, task_to_run
	if len(dest_path) > 4096:
		print(f'\nSkipped {dest_path} because path is too long')
		return 0, 0, symLinks #, task_to_run
	startTime = time.perf_counter()
	try:
		src_size = os.path.getsize(src_path)
		copiedSize = 0
		
		if os.path.exists(dest_path) and (not os.path.islink(src_path)) and is_file_identical(src_path, dest_path,src_size,full_hash):
			# if verbose:
			#     print(f'\nSkipped {src_path}')
			st = os.stat(src_path,follow_symlinks=False)
			# src_size = stis_file.st_rsize if 'st_rsize' in st else st.st_blocks * 512
			shutil.copystat(src_path, dest_path,follow_symlinks=False)
			if os.name == 'posix':
				os.chown(dest_path, st.st_uid, st.st_gid)
				#also copy the modes
				os.chmod(dest_path, st.st_mode)
			os.utime(dest_path, (st.st_atime, st.st_mtime),follow_symlinks=False)
			endTime = time.perf_counter()
			return 0, endTime - startTime , symLinks #, task_to_run
		if os.path.islink(src_path):
			symLinks[src_path] = dest_path
		else:
			try:
				if os.name == 'posix':

					run_command_in_multicmd_with_path_check(["cp", "-af", "--sparse=always", src_path, dest_path],timeout=0,quiet=True)
					#task_to_run = ["cp", "-af", "--sparse=always", src_path, dest_path]
					st = os.stat(dest_path,follow_symlinks=False)
					copiedSize = st.st_rsize if 'st_rsize' in st else st.st_blocks * 512
				else:
					shutil.copy2(src_path, dest_path, follow_symlinks=False)
					#shutil.copystat(src_path, dest_path)
			except Exception as e:
				import traceback
				print(f'\nError copying {src_path} to {dest_path}: {e}')
				print(traceback.format_exc())
				if os.name == 'posix':
					run_command_in_multicmd_with_path_check(["cp", "-af", src_path, dest_path],timeout=0,quiet=True)
					#task_to_run = ["cp", "-af", src_path, dest_path]
				elif os.name == 'nt':
					run_command_in_multicmd_with_path_check(["xcopy", "/I", "/E", "/Y", "/c", "/q", "/k", "/r", "/h", "/x", src_path, dest_path],timeout=0,quiet=True)
					#task_to_run = ["xcopy", "/I", "/E", "/Y", "/c", "/q", "/k", "/r", "/h", "/x", src_path, dest_path]
	except Exception as e:
		print(f'\nError copying {src_path} to {dest_path}: {e}')
		return 0, time.perf_counter() - startTime, symLinks #, task_to_run
	if not copiedSize:
		copiedSize = src_size
	endTime = time.perf_counter()
	return copiedSize, endTime - startTime , symLinks #, task_to_run

def copy_files_bulk(src_files, dst_files, full_hash=False, verbose=False):
	"""
	Copy multiple files from source to destination.

	Args:
		src_files (list): List of source file paths.
		dst_files (list): List of destination file paths.
		full_hash (bool, optional): Whether to calculate full hash of files. Defaults to False.
		verbose (bool, optional): Whether to display verbose output. Defaults to False.

	Returns:
		tuple: A tuple containing the total size of copied files, total time taken for copying, and a dictionary of symbolic links.

	"""
	total_size = 0
	total_time = 0
	symLinks = {}
	# tasks_to_run = []
	for src, dst in zip(src_files, dst_files):
		#total_size += copy_file(src, dst, full_hash, verbose)
		size , cpTime , rtnSymLinks = copy_file(src, dst, full_hash, verbose)
		# if task_to_run:
		#     tasks_to_run.append(task_to_run)
		total_size += size
		total_time += cpTime
		symLinks.update(rtnSymLinks)
	# startTime = time.perf_counter()
	# if tasks_to_run:
	#     multiCMD.run_commands(tasks_to_run, timeout=0,max_threads=3, quiet=True)
	# endTime = time.perf_counter()
	# total_time += endTime - startTime
	return total_size , total_time, symLinks

def copy_file_list_parallel(file_list, links, src_path, dest_path, max_workers, full_hash=False, verbose=False, files_per_job=1,estimated_size = 0):
	"""
	Copy a list of files in parallel using multiple workers.

	Args:
		file_list (list): List of file paths to be copied.
		links (list): List of symbolic links to be created.
		src_path (str): Source directory path.
		dest_path (str): Destination directory path.
		max_workers (int): Maximum number of worker processes to use.
		full_hash (bool, optional): Whether to perform full file hash comparison. Defaults to False.
		verbose (bool, optional): Whether to print verbose output. Defaults to False.
		files_per_job (int, optional): Number of files to be copied per job. Defaults to 1.

	Returns:
		tuple: A tuple containing the following information:
			- copy_counter (int): Total number of files copied.
			- copy_size_counter (int): Total size of files copied in bytes.
			- symLinks (dict): Dictionary mapping symbolic links to their destination paths.
			- file_list (frozenset): Frozen set of remaining files to be copied.
	"""
	if len(src_path) > 4096:
		print(f'\nSkipped {src_path} because path is too long')
		return 0, 0, {}, frozenset()
	if len(dest_path) > 4096:
		print(f'\nSkipped {dest_path} because path is too long')
		return 0, 0, {}, frozenset()
	file_list_iterator = iter(file_list)
	startTime = time.perf_counter()
	lastRefreshTime = startTime
	total_files = len(file_list)
	futures = {}
	files_per_job = max(1,files_per_job)
	symLinks = {}
	for link in links:
		symLinks[link] = os.path.join(dest_path, os.path.relpath(link, src_path))
	if len(file_list) == 0:
		return 0, 0, symLinks , frozenset()
	print(f"Processing {len(file_list)} files with {max_workers} workers")
	apb = Adaptive_Progress_Bar(total_count=total_files,total_size=estimated_size,last_num_job_for_stats=max(1,max_workers//10),process_word='Copied',use_print_thread = True)
	with ProcessPoolExecutor(max_workers=max_workers) as executor:
		while file_list_iterator or futures:
			# counter = 0
			while file_list_iterator and len(futures) < 1.1 * max_workers and time.perf_counter() - lastRefreshTime < 1:
				src_files = []
				dst_files = []
				try:
					# generate some noise from 0.9 to 1.1 to apply to the files per job to attempt spreading out the job scheduling
					noise = random.uniform(0.9, 1.1)
					for _ in range(max(1,round(files_per_job * noise))):
						src_file = next(file_list_iterator)
						dst_file = os.path.join(dest_path, os.path.relpath(src_file, src_path))
						src_files.append(src_file)
						dst_files.append(dst_file)
					future = executor.submit(copy_files_bulk, src_files, dst_files, full_hash, verbose)
					futures[future] = src_files
					# counter += 1
				except StopIteration:
					if src_files:
						future = executor.submit(copy_files_bulk, src_files, dst_files, full_hash, verbose)
						futures[future] = src_files
					file_list_iterator = None

			done, _ = concurrent.futures.wait(futures.keys(), return_when=concurrent.futures.FIRST_COMPLETED)

			#print('\n',len(done) ,'\t', len(futures))
			if file_list_iterator and len(done) > 1 and len(done) / len(futures) > 0.1:
				if verbose:
					print(f'\nTransfer is fast, doubling files per job to {files_per_job * 2}')
				files_per_job *= 2

			current_iteration_total_run_time = 0
			for future in done:
				src_files = futures.pop(future)
				copied_file_count_this_run = len(src_files)
				#try:
				cpSize, cpTime, rtnSymLinks = future.result()
				current_iteration_total_run_time += cpTime
				apb.update(num_files=copied_file_count_this_run,cpSize=cpSize,cpTime=cpTime,files_per_job=files_per_job)
				apb.scheduled_jobs = len(futures)
				symLinks.update(rtnSymLinks)
			if verbose:
					print(f'\nAverage cptime is {current_iteration_total_run_time / len(done):0.2f} for {len(done)} jobs with {copied_file_count_this_run} files each')

			if file_list_iterator and copied_file_count_this_run == files_per_job and time.perf_counter() - lastRefreshTime > 1:
				files_per_job //= 1.61803398875
				files_per_job = round(files_per_job)
				if verbose:
					print(f'\nCompletion time is long, changing files per job to {files_per_job}')
			#elif file_list_iterator and copied_file_count_this_run == files_per_job and current_iteration_total_run_time / len(done) < 1:
			elif file_list_iterator and copied_file_count_this_run == files_per_job and time.perf_counter() - lastRefreshTime < 0.1:
				files_per_job *= 1.61803398875
				files_per_job = round(files_per_job)
				if verbose:
					print(f'\nCompletion time is short, changing files per job to {files_per_job}')
			if files_per_job < 1:
				files_per_job = 1
			lastRefreshTime = time.perf_counter()
	endTime = time.perf_counter()
	apb.stop()
	print(f"\nTime taken:             {endTime-startTime:0.4f} seconds")
	print(f"Average speed:          {format_bytes(apb.size_counter / (endTime-startTime))}B/s")
	print(f"                        {apb.item_counter / (endTime-startTime):.2f} file/s")
	print(f"Average bandwidth:      {format_bytes(apb.size_counter / (endTime-startTime) * 8,use_1024_bytes=False)}bps")

	return apb.item_counter, apb.size_counter , symLinks ,file_list

def copy_files_parallel(src_path, dest_path, max_workers, full_hash=False, verbose=False,files_per_job=1,parallel_file_listing=False,exclude=None):
	# skip if src path or dest path is longer than 4096 characters
	if len(src_path) > 4096:
		print(f'Skipping: {src_path} is too long')
		return 0, 0 , set(), frozenset()
	if len(dest_path) > 4096:
		print(f'Skipping: {dest_path} is too long')
		return 0, 0 , set(), frozenset()
	
	if exclude and is_excluded(src_path,exclude):
		return 0, 0 , set(), frozenset()

	if not os.path.isdir(src_path):
		src_size, _ , symLinks = copy_file(src_path, dest_path,full_hash=full_hash, verbose=verbose)
		return 1, src_size , symLinks , frozenset([src_path])
	startTime = time.perf_counter()
	if parallel_file_listing:
		file_list , links,init_size,folders  = get_file_list_parallel(src_path, max_workers,exclude=exclude)
	else:
		file_list,links,init_size,folders = get_file_list_serial(src_path,exclude=exclude)
		
	endTime = time.perf_counter()
	print(f"Time taken to get file list: {endTime-startTime:0.4f} seconds")
	total_files = len(file_list)
	print(f"Number of files: {total_files}")
	print(f"Number of links: {len(links)}")
	print(f"Number of folders: {len(folders)}")
	print(f"Estimated size: {format_bytes(init_size)}B")
	return copy_file_list_parallel(file_list=file_list,links=links,src_path=src_path, dest_path=dest_path, max_workers=max_workers, full_hash=full_hash, verbose=verbose,files_per_job=files_per_job,estimated_size = init_size)

def copy_files_serial(src_path, dest_path, full_hash=False, verbose=False,exclude=None):
	# skip if src path or dest path is longer than 4096 characters
	if len(src_path) > 4096:
		print(f'Skipping: {src_path} is too long')
		return 0, 0 , set(), frozenset()
	if len(dest_path) > 4096:
		print(f'Skipping: {dest_path} is too long')
		return 0, 0 , set(), frozenset()
	if exclude and is_excluded(src_path,exclude):
		return 0, 0 , set(), frozenset()
	if not os.path.isdir(src_path):
		src_size, _ , symLinks = copy_file(src_path, dest_path,full_hash=full_hash, verbose=verbose)
		return 1, src_size , symLinks , frozenset([src_path])
	print(f'Getting file list for {src_path}')
	file_list,links,init_size,folders = get_file_list_serial(src_path,exclude=exclude)
	links = set(links)
	total_files = len(file_list)
	print(f"Number of files: {total_files}")
	startTime = time.perf_counter()
	apb = Adaptive_Progress_Bar(total_count=total_files,total_size=init_size,last_num_job_for_stats=1,process_word='Copied')
	for file in file_list:
		size, cpTime ,rtnSymLinks = copy_file(file, os.path.join(dest_path, os.path.relpath(file, src_path)),full_hash = full_hash, verbose=verbose)
		#update_progress_bar(copy_counter, copy_size_counter, total_files, startTime)
		apb.update(num_files=1,cpSize=size,cpTime=cpTime,files_per_job=1)
		links.update(rtnSymLinks)
	symLinks = {}
	for link in links:
		symLinks[link] = os.path.join(dest_path, os.path.relpath(link, src_path))
	endTime = time.perf_counter()
	apb.stop()
	print(f"\nTime taken:             {endTime-startTime:0.4f} seconds")
	print(f"Average speed:          {format_bytes(apb.size_counter / (endTime-startTime))}B/s")
	print(f"                        {apb.item_counter / (endTime-startTime):.2f} file/s")
	print(f"Average bandwidth:      {format_bytes(apb.size_counter / (endTime-startTime) * 8,use_1024_bytes=False)}bps")
	return apb.item_counter, apb.size_counter , symLinks , frozenset(file_list)

# ---- Copy Directories ----
def sync_directory_metadata(src_path, dest_path):
	# skip if src path or dest path is longer than 4096 characters
	if len(src_path) > 4096:
		print(f'Skipping: {src_path} is too long')
		return 0, 0 , set(), frozenset()
	if len(dest_path) > 4096:
		print(f'Skipping: {dest_path} is too long')
		return 0, 0 , set(), frozenset()
	startTime = time.perf_counter()
	if os.path.islink(src_path):
		return 0,time.perf_counter()-startTime,{src_path: dest_path}
	if not os.path.isdir(src_path):
		return copy_file(src_path, dest_path)
	# Create the directory if it does not exist
	try:
		if not (os.path.exists(dest_path) or os.path.ismount(dest_path)):
			os.makedirs(dest_path, exist_ok=True)
	except FileExistsError as e:
		print(f"Destination path {dest_path} maybe a mounted dir, known issue with os.path.exists\nContinuing without creating dest folder...")

	# Sync the metadata
	shutil.copystat(src_path, dest_path)
	# run_command_in_multicmd_with_path_check(["cp",'--no-dereference','--preserve=all',src_path,dest_path],timeout=0,quiet=True)
	st = os.stat(src_path)
	if os.name == 'posix':
		os.chown(dest_path, st.st_uid, st.st_gid)
	os.utime(dest_path, (st.st_atime, st.st_mtime))
	return 1,time.perf_counter()-startTime,frozenset()

def sync_directory_metadata_bulk(src_paths, dest_paths):
	total_count = 0
	total_time = 0
	symLinks = {}
	for src, dst in zip(src_paths, dest_paths):
		#total_size += copy_file(src, dst, full_hash, verbose)
		count , cpTime , rtnSymLinks = sync_directory_metadata(src, dst)
		total_count += count
		total_time += cpTime
		symLinks.update(rtnSymLinks)
	return total_count , total_time, symLinks
		
def sync_directories_serial(src, dest,exclude=None):
	"""
	Synchronizes directories from source to destination in a single thread.

	Args:
		src (str): The source directory path.
		dest (str): The destination directory path.
		exclude (list, optional): A list of patterns to exclude from synchronization.

	Returns:
		tuple: A tuple containing:
			- int: Number of directories synced (always 0 in this implementation).
			- int: Total size of directories synced in bytes (always 0 in this implementation).
			- set: Set of synchronized directories (always empty in this implementation).
			- frozenset: Set of symbolic links (always empty in this implementation).

	Notes:
		- If the source or destination path length exceeds 4096 characters, the function will skip synchronization.
		- If the source path is excluded based on the exclude patterns, the function will skip synchronization.
		- The function prints progress information to the terminal, including the number of directories synced and the speed of synchronization.
	"""
	# skip if src path or dest path is longer than 4096 characters
	if len(src) > 4096:
		print(f'Skipping: {src} is too long')
		return 0, 0 , set(), frozenset()
	if len(dest) > 4096:
		print(f'Skipping: {dest} is too long')
		return 0, 0 , set(), frozenset()
	if exclude and is_excluded(src,exclude):
		return 0, 0 , set(), frozenset()
	symLinks = {}
	if not os.path.isdir(src):
		_, _ , symLinks = copy_file(src, dest)
		return symLinks
	#sync_directory_metadata(src, dest)
	print(f'Getting file list for {src}')
	file_list,links,init_size,folders = get_file_list_serial(src,exclude=exclude)
	print(f"Syncing Dir from {src} to {dest} in single thread")
	apb = Adaptive_Progress_Bar(total_count=len(folders),total_size=len(folders))
	for folder in folders:
		count , cpTime , rtnSymLinks = sync_directory_metadata(folder, os.path.join(dest, os.path.relpath(folder, src)))
		apb.update(num_files=1, cpSize=count, cpTime=cpTime , files_per_job=1)
	apb.stop()
	# try:
	# 	columns, _ = os.get_terminal_size()
	# except :
	# 	columns = 80
	# for root, dirs, _ in os.walk(src, topdown=True):
	# 	for dir in dirs:
	# 		src_path = os.path.join(root, dir)
	# 		if exclude and is_excluded(src_path,exclude):
	# 			continue
	# 		dest_path = os.path.join(dest, os.path.relpath(src_path, src))
	# 		returnCount,_,returnLinks = sync_directory_metadata(src_path, dest_path)
	# 		symLinks.update(returnLinks)
	# 		count += 1
	# 		copiedFolderCount += returnCount
	# 		# print the count and speed in Directory/s
	# 		outStr = f"\r{count} ({format_bytes(copiedFolderCount)}B) directories synced, {count / (time.perf_counter() - startTime):.2f} directories/s Copied {dir}"
	# 		# we fill and truncate the string to prevent artifacts from previous prints
	# 		sys.stdout.write(outStr.ljust(columns)[:columns])
	# 		sys.stdout.flush()
	return symLinks

# def get_all_folders_iter(path,exclude=None):
# 	for entry in os.scandir(path):
# 		if entry.is_dir():
# 			if exclude and is_excluded(entry.path,exclude):
# 				continue
# 			yield entry.path
# 			# check if the folder is a symlink and if it is too long
# 			if not entry.is_symlink():
# 				if len(entry.path) < 4096:
# 					yield from get_all_folders_iter(entry.path,exclude=exclude)
# 				else:
# 					print(f"Skipping {entry.path} as it is too long !!!!!!!!!!!!!!")

def sync_directories_parallel(src, dest, max_workers, verbose=False,folder_per_job=64,exclude=None):
	# skip if src path or dest path is longer than 4096 characters
	if len(src) > 4096:
		print(f'Skipping: {src} is too long')
		return 0, 0 , set(), frozenset()
	if len(dest) > 4096:
		print(f'Skipping: {dest} is too long')
		return 0, 0 , set(), frozenset()
	symLinks = {}
	if exclude and is_excluded(src,exclude):
		return 0, 0 , set(), frozenset()
	# 
	if not os.path.isdir(src):
		_, _ , symLinks = copy_file(src, dest)
		return symLinks
	sync_directory_metadata(src, dest)
	print(f'Getting file list for {src}')
	file_list,links,init_size,folders = get_file_list_serial(src,exclude=exclude)
	folder_list_iterator = iter(folders)
	futures = {}
	startTime = time.perf_counter()
	last_refresh_time = startTime
	max_workers = max(2,int(max_workers / 4))

	folder_per_job = max(1,folder_per_job)
	num_folders_copied_this_job = 0

	print(f"Syncing Dir from {src} to {dest} with {max_workers} workers")
	apb = Adaptive_Progress_Bar(total_count=len(folders),total_size=len(folders),use_print_thread = True)
	with ProcessPoolExecutor(max_workers=max_workers) as executor:
		while folder_list_iterator or futures:
			# counter = 0
			while folder_list_iterator and len(futures) <  max_workers and last_refresh_time - time.perf_counter() < 5:
				src_folders = []
				dst_folders = []
				try:
					for _ in range(folder_per_job):
						src_folder = next(folder_list_iterator)
						dst_folder = os.path.join(dest, os.path.relpath(src_folder, src))
						src_folders.append(src_folder)
						dst_folders.append(dst_folder)
					future = executor.submit(sync_directory_metadata_bulk, src_folders, dst_folders)
					futures[future] = src_folders
					# counter += 1
				except StopIteration:
					if src_folders:
						future = executor.submit(sync_directory_metadata_bulk, src_folders, dst_folders)
						futures[future] = src_folders
					folder_list_iterator = None

			done, _ = concurrent.futures.wait(futures.keys(), return_when=concurrent.futures.FIRST_COMPLETED)
			time_spent_this_iter = 0
			for future in done:
				src_folders = futures.pop(future)
				num_folders_copied_this_job = len(src_folders)
				#try:
				cpSize, cpTime, rtnSymLinks = future.result()
				time_spent_this_iter += cpTime
				#except Exception as exc:
					#print(f'\n{future} generated an exception: {exc}')
				#else:
				apb.update(num_files=num_folders_copied_this_job, cpSize=cpSize, cpTime=cpTime , files_per_job=folder_per_job)
				apb.scheduled_jobs = len(futures)
				symLinks.update(rtnSymLinks)
			if verbose:
					print(f'\nAverage cptime is {time_spent_this_iter / len(done):0.2f} for {len(done)} jobs with {num_folders_copied_this_job} folders each')

			if folder_list_iterator and num_folders_copied_this_job == folder_per_job and (time_spent_this_iter / len(done) > 5 or time.perf_counter() - last_refresh_time > 5):
				folder_per_job //= 1.61803398875
				folder_per_job = round(folder_per_job)
				if verbose:
					print(f'\nCompletion time is long, changing folders per job to {folder_per_job}')
			elif folder_list_iterator and num_folders_copied_this_job == folder_per_job and time_spent_this_iter / len(done) < 1:
				folder_per_job *= 1.61803398875
				folder_per_job = round(folder_per_job)
				if verbose:
					print(f'\nCompletion time is short, changing folders per job to {folder_per_job}')
			if folder_per_job < 1:
				folder_per_job = 1
			last_refresh_time = time.perf_counter()

	endTime = time.perf_counter()
	apb.stop()
	print(f"\nTime taken:             {endTime-startTime:0.4f} seconds")
	print(f"Average speed:          {format_bytes(apb.size_counter / (endTime-startTime))}B/s")
	print(f"                        {apb.item_counter / (endTime-startTime):.2f} folder/s")
	print(f"Average bandwidth:      {format_bytes(apb.size_counter / (endTime-startTime) * 8,use_1024_bytes=False)}bps")
	return symLinks

# ---- Compare Files ----
def compareFileList(file_list, file_list2,diff_file_list=None,tar_diff_file_list = False):
	print('-'*80)
	print(f"Number of files in src: {len(file_list)}")
	print(f"Number of files in dest: {len(file_list2)}")
	# Now we print out a detailed comparison
	print('-'*80)
	inSrcNotInDest = file_list - file_list2
	inDestNotInSrc = file_list2 - file_list
	print(f"Files in src but not in dest:")
	for file in inSrcNotInDest:
		print(file)
	print('-'*80)
	print(f"Files in dest but not in src:")
	for file in inDestNotInSrc:
		print(file)
	print('-'*80)
	print(f"Files in src but not in dest count: {len(inSrcNotInDest)}")
	print(f"Files in dest but not in src count: {len(inDestNotInSrc)}")
	if diff_file_list:
		with open(diff_file_list,'w') as f:
			if inDestNotInSrc:
				for file in inDestNotInSrc:
					f.write((file.rpartition(':')[0] if ':' in file else file)+'\n')
			if inSrcNotInDest and not tar_diff_file_list:
				#f.write('-\0'+'\n-\0'.join(inSrcNotInDest.rpartition(':')[0] if ':' in inSrcNotInDest else inSrcNotInDest)+ '\n') 
				for file in inSrcNotInDest:
					f.write('-\0'+(file.rpartition(':')[0] if ':' in file else file)+'\n')
		print(f"Diff file list written to {diff_file_list}")

# ---- Remove Extra ----
def remove_extra_dirs(src_paths, dest,exclude=None):
	"""
	Removes extra directories in destination that are not present in the source paths.

	:param src_paths: list of source paths
	:param dest: destination path
	:return: None
	"""
	# Skip if the dest path is too long
	if len(dest) > 4096:
		print(f"Skipping {dest} because the path is too long")
		return
	if exclude and is_excluded(dest,exclude):
		return
	# remove excluded paths from src_paths
	for src_path in src_paths:
		if exclude and is_excluded(src_path,exclude):
			src_paths.remove(src_path)


	extraDirs = []
	for dirpath, dirnames, _ in os.walk(dest, topdown=False):
		for dirname in dirnames:
			if not dirname.endswith(os.path.sep):
				dirname += os.path.sep
			dest_dir_path = os.path.join(dirpath, dirname)
			# Check if the directory exists in the source paths
			if not any(os.path.exists(os.path.join(os.path.dirname(src_path), os.path.relpath(dest_dir_path, dest))) for src_path in src_paths):
				if exclude and not is_excluded(dest_dir_path,exclude):
					print(f"Deleting extra directory: {dest_dir_path}")
					extraDirs.append(dest_dir_path)
	for dir in extraDirs:
		os.rmdir(dir)

def remove_extra_files(total_file_list, dest,max_workers,verbose,files_per_job,single_thread=False,exclude=None):
		print(f"Removing extra files from {dest} with {max_workers} workers")
		# we first get a file list of the dest dir
		if single_thread:
			dest_file_list,links,init_size, folders = get_file_list_serial(dest,exclude=exclude)
		else:
			dest_file_list,links,init_size ,folders = get_file_list_parallel(dest, max_workers,exclude=exclude)
		dest_file_list = trimPaths(dest_file_list,dest)
		dest_file_list.update(trimPaths(links,dest))
		# we then get the list of all extra files
		inDestNotInSrc = dest_file_list - total_file_list
		inDestNotInSrc = [os.path.join(dest,file) for file in inDestNotInSrc]
		print('-'*80)
		print(f"Files in dest but not in src:")
		for file in inDestNotInSrc:
			print(file)
		print('-'*80)
		if len(inDestNotInSrc) == 0:
			print(f"No extra files found in {dest}")
		else:
			print(f"Files in dest but not in src count: {len(inDestNotInSrc)}")
			print(f"Do you want to delete them? (y/n)")
			if not input().lower().startswith('y'):
				exit(0)
			startTime = time.perf_counter()
			if single_thread:
				for file in inDestNotInSrc:
					if os.path.isfile(file) or os.path.islink(file):
						os.remove(file)
					else:
						shutil.rmtree(file)
			else:
				delete_file_list_parallel(inDestNotInSrc, max_workers, verbose,files_per_job)
			endTime = time.perf_counter()
			print(f"Time taken to remove extra files: {endTime-startTime:0.4f} seconds")

# ---- Main Helper Functions ----
def mountSrcImage(src_image,src_images: list,src_paths: list,mount_points: list,loop_devices: list):
	for src_image_pattern in src_image:
		if not os.name == 'nt':
			try:
				src_images.extend(glob.glob(src_image_pattern,include_hidden=True,recursive=True))
			except:
				src_images.extend(glob.glob(src_image_pattern,recursive=True))
		else:
			src_images.append(src_image_pattern)
	src_str = ''
	for src in src_images:
		if not os.path.exists(src):
			print(f"Source image {src} does not exist")
			src_images.remove(src)
		src_str += f"{os.path.basename(src)}-"
		# we will mount the all image all partitions to seperate temorary folders and add them to src_paths
		loop_device_dest = create_loop_device(src,read_only=True)
		loop_devices.append(loop_device_dest)
		partitions = get_partitions(loop_device_dest)
		if not partitions:
			# if there are no partitions, we mount the image directly
			partitions = [loop_device_dest]
		for partition in partitions:
			try:
				target_mount_point = tempfile.mkdtemp()
				mount_points.append(target_mount_point)
				print(f"Mounting {partition} at {target_mount_point}")
				run_command_in_multicmd_with_path_check(["mount","-o","ro",partition,target_mount_point])
				# verify mount 
				if os.path.ismount(target_mount_point):
					src_paths.append(target_mount_point + os.path.sep)
				else:
					print(f"Partition {partition} cannot be mounted, what to do? (s/f/n)")
					print(f"s:  Skip {partition} \t:Skip this partition and continue mounting the rest ( default )")
					print(f"f:  Fix {partition} \t:Try to fix the partition and mount it again")
					print(f"n:  Exit")
					# Wait for user input with a 5 second timeout
					inStr = multiCMD.input_with_timeout_and_countdown(5)
					if (not inStr) or inStr.lower().startswith('s'):
						print(f"Partition {partition} cannot be mounted, skipping")
						continue
					elif inStr.lower().startswith('f'):
						fix_fs(partition)
						run_command_in_multicmd_with_path_check(["mount","-o","ro",partition,target_mount_point])
						if os.path.ismount(target_mount_point):
							src_paths.append(target_mount_point + os.path.sep)
						else:
							print(f"Partition {partition} cannot be mounted after fixing, skipping")
							continue
					else:
						print(f"Exiting")
						exit(0)
			except Exception as e:
				print(f"Error mounting partition {partition}, skipping")
				print(e)
				continue
	return src_str.strip('-')

def verifySrcPath(src_path,src_paths: list):
	if src_path:
		for src_path_pattern in src_path:
			#print(src_path_pattern)
			if not os.name == 'nt':
				try:
					src_paths.extend(glob.glob(src_path_pattern,include_hidden=True,recursive=True))
				except:
					src_paths.extend(glob.glob(src_path_pattern,recursive=True))
			else:
				src_paths.append(src_path_pattern)
			#print(src_paths)
	for src in src_paths:
		if not os.path.exists(src):
			print(f"Source path {src} does not exist")
			src_paths.remove(src)
		# if ':' in src:
		#     print(f"Remote syncing is not supported in this version, removing source path {src}.")
		#     src_paths.remove(src)
	if len(src_paths) == 0:
		print(f"No source paths specified, exiting")
		exit(0)

def loadFileList(file_list):
	if not os.path.exists(file_list):
		print(f"File list {file_list} does not exist")
		return frozenset()
	with open(file_list, 'r') as f:
		fileList = frozenset([entry.strip() for entry in f.read().splitlines() if entry.strip()])
	return fileList

def storeFileList(file_list, src_paths: list, single_thread=False, max_workers=4 * multiprocessing.cpu_count(), verbose=False,
					files_per_job=1, compare_file_list=False, remove_extra=False, parallel_file_listing=False, exclude=None,
					diff_file_list=None,tar_diff_file_list = False, src_str=None,append_hash=True,full_hash=False):
	"""
	Process a file list by performing various operations such as removing extra files, getting file lists from source paths,
	comparing file lists, and writing the final file list to a file.

	Args:
		file_list (str): The path to the file list.
		src_paths (list): A list of source paths from which to get the file list.
		single_thread (bool, optional): Whether to use a single thread for file operations. Defaults to False.
		max_workers (int, optional): The maximum number of worker processes to use for parallel file listing. Defaults to 4 times the number of CPU cores.
		verbose (bool, optional): Whether to print verbose output. Defaults to False.
		files_per_job (int, optional): The number of files to process per job when removing extra files. Defaults to 1.
		compare_file_list (bool, optional): Whether to compare the file list with the source paths. Defaults to False.
		remove_extra (bool, optional): Whether to remove extra files from the file list. Defaults to False.
		parallel_file_listing (bool, optional): Whether to use parallel file listing. Defaults to False.
		exclude (str, optional): A pattern to exclude files from the file list. Defaults to None.

	Returns:
		None
	"""

	if os.name == 'nt' and not file_list.endswith('_file_list.txt'):
		file_list += '_file_list.txt'
	if remove_extra:
		print('-' * 80)
		file_list_file = loadFileList(file_list)
		if len(src_paths) == 1:
			remove_extra_files(file_list_file, src_paths[0], max_workers, verbose, files_per_job, single_thread, exclude=exclude)
			print('-' * 80)
		else:
			print("Currently only supports removing extra files for a single src_path when using file_list")
		exit(0)
	fileList = set()
	for src in src_paths:
		print(f"Getting file list from {src}")
		startTime = time.perf_counter()
		if not parallel_file_listing:
			files, links, init_size,folders = get_file_list_serial(src, exclude=exclude,append_hash=append_hash,full_hash=full_hash)
		else:
			files, links, init_size,folders  = get_file_list_parallel(src, max_workers, exclude=exclude,append_hash=append_hash,full_hash=full_hash)
		fileList.update(trimPaths(files, src))
		fileList.update(trimPaths(links, src))
		fileList.update([folder_path + os.path.sep for folder_path in trimPaths(folders, src)])
		endTime = time.perf_counter()
		print(f"Time taken to get file list: {endTime - startTime:0.4f} seconds")
	if compare_file_list:
		# This means we have a file_list and a src_path so we compare them
		print(f"Comparing file list from {src_paths} with {file_list}")
		if diff_file_list == 'auto':
			if not src_str:
				src_str = '-'.join([os.path.basename(os.path.realpath(src)) for src in src_paths])
			diff_file_list = f'DIFF_{src_str}_TO_{os.path.basename(os.path.realpath(file_list))}_{int(time.time())}_{"tar_" if tar_diff_file_list else ""}file_list.txt'
		fileList2 = loadFileList(file_list)
		compareFileList(fileList, fileList2,diff_file_list,tar_diff_file_list = tar_diff_file_list)
	else:
		print(f"Number of files: {len(fileList)}")
		print(f"Writing file list to {fileList}")
		# sort the file list
		fileList = natural_sort(fileList)
		with open(file_list, 'w') as f:
			for file in fileList:
				f.write(file + '\n')
	# cleanUp(mount_points,loop_devices)
	# exit(0)

def processRemove(src_paths: list,single_thread = False, max_workers = 4 * multiprocessing.cpu_count(),verbose = False,
				  files_per_job = 1, remove_force = False,exclude=None,batch = False):
	print(f"Removing files from {src_paths} with {max_workers if not single_thread else '1'} workers")
	#src = os.path.abspath(src +os.path.sep)
	src_paths = [os.path.abspath(src + os.path.sep) for src in src_paths if os.path.exists(src)]
	if not remove_force:
		print(f"Do you want to continue? (y/n)")
		if not input().lower().startswith('y'):
			exit(0)
	processedPaths = []
	for path in src_paths:
		while os.path.basename(path) == '':
			path = os.path.dirname(path)
		processedPaths.append(path)
	for path in processedPaths:
		print('-'*80)
		startTime = time.perf_counter()
		if single_thread:
			shutil.rmtree(path)
		else:
			delete_files_parallel(processedPaths if batch else path, max_workers, verbose=verbose,files_per_job=files_per_job,exclude=exclude,batch=batch)
		endTime = time.perf_counter()
		print(f"Time taken to remove files: {endTime-startTime:0.4f} seconds")
		print('-'*80)
		if batch and not single_thread:
			break

def getDestFromImage(dest_image,mount_points: list,loop_devices: list):
	dest = ''
	target_mount_point = tempfile.mkdtemp()
	mount_points.append(target_mount_point)
	# see if dest_image file exist.
	# if it does, then we will setup a loop device and attempt to mount it.
	if os.path.exists(dest_image):
		print(f"Destination image {dest_image} exists, attempting to mount it.")
		# setup a loop device
		target_loop_device_dest = create_loop_device(dest_image)
		loop_devices.append(target_loop_device_dest)
		target_partition = get_largest_partition(target_loop_device_dest)
		# mount the loop device to a temporary folder
		print(f"Mounting {target_partition} at {target_mount_point}")
		run_command_in_multicmd_with_path_check(["mount",target_partition,target_mount_point])
		# verify mount 
		if os.path.ismount(target_mount_point):
			dest = target_mount_point + os.path.sep
		else:
			print(f"Destination image cannot be mounted, do you want to continue? (f/d/n)")
			print(f"f:  Fix {target_partition} \t:Try to fix the partition and mount it again ( default )")
			print(f"d:  Delete {dest_image} \t:Delete the old and create a new image (Warning: this will overwrite the existing image)")
			print(f"n:  Exit")
			inStr = multiCMD.input_with_timeout_and_countdown(15)
			if (not inStr) or inStr.lower().startswith('f'):
				try:
					# First find out the fs type
					fs_type = get_fs_type(target_partition)
					fix_fs(target_partition,fs_type)
					# mount it again
					run_command_in_multicmd_with_path_check(["mount",target_partition,target_mount_point])
					# verify mount
					if os.path.ismount(target_mount_point):
						dest = target_mount_point + os.path.sep
					else:
						print(f"Destination image cannot be mounted after fixing, exiting")
						exit(0)
				except Exception as e:
					print(f"Error fixing {target_partition}, exiting")
					print(e)
					exit(0)
			elif inStr.lower().startswith('d'):
				run_command_in_multicmd_with_path_check(['losetup','-d',target_loop_device_dest])
				delete_file_bulk([dest_image])
				dest = None
			else:
				exit(0)
	else:
		dest = None
	return dest , target_mount_point

def getDestFromPath(dest_path,src_paths: list,src_path,can_be_none = False):
	dest = ''
	src_path = list(src_path)
	if not dest_path:
		if can_be_none:
			return None
		if len(src_path) > 1:
			try:
				cwd = os.path.join(os.getcwd()) + os.path.sep
			except:
				cwd = None
			print(f"Destination path not specified, do you want to continue? (l/y/n/...)")
			print(f"l:  {src_path[-1]} \t:Use last src_path in list ( default )")
			if cwd:
				print(f"y:  {cwd} \t:Use current working directory")
			print(f"n:  Exit")
			print(f"...:  Enter custom destination path")
			inStr = multiCMD.input_with_timeout_and_countdown(60)
			if (not inStr) or inStr.lower() == 'l':
				dest = src_path[-1]
				src_paths.remove(dest) if dest in src_paths else None
				print(f"Destination path not specified, using {dest}")
			elif cwd and inStr.lower() == 'y':
				dest = cwd
				print(f"Destination path not specified, using {dest}")
			elif inStr.lower() == 'n':
				exit(0)
			else:
				dest = inStr
		else:
			try:
				cwd = os.path.join(os.getcwd()) + os.path.sep
			except:
				cwd = None
			print(f"Destination path not specified, src_path length is 1, do you want to continue? (y/n/...)")
			if cwd:
				print(f"y:  {os.getcwd() + os.path.sep} \t:Use current working directory ( default )")
			print(f"n:  Exit")
			print(f"...:  Enter custom destination path")
			inStr = multiCMD.input_with_timeout_and_countdown(60)
			if (not inStr) or cwd and inStr.lower() == 'y':
				dest = cwd
				print(f"Destination path not specified, using {dest}")
			elif inStr.lower() == 'n':
				exit(0)
			else:
				dest = inStr
	else:
		dest = dest_path
	# if ':' in dest:
	#     print(f"Remote syncing is not supported in this version, destination {dest} include ':', exiting.")
	#     exit(0)
	if len(src_paths) == 1 and os.path.isdir(src_paths[0]) and not src_paths[0].endswith(os.path.sep) and not dest.endswith(os.path.sep):
		src_paths[0] += os.path.sep
		dest += os.path.sep
	if (len(src_paths) > 1 or src_paths[0].endswith(os.path.sep)) and not dest.endswith(os.path.sep):
		dest += os.path.sep
	return dest

def processCompareFileList(src_paths: list, dest, max_workers = 4 * multiprocessing.cpu_count(),
						   file_list = "",parallel_file_listing = False,exclude=None,dest_image = None,diff_file_list = None,tar_diff_file_list = False,
						   append_hash = True,full_hash = False):
	# while os.path.basename(dest) == '':
	#     dest = os.path.dirname(dest)
	if not dest or not os.path.exists(dest) or not os.path.isdir(dest):
		print(f"Destination image {dest_image} does not exist or dest {dest} is empty, exiting.")
		exit(1)
	print(f"Comparing file list from {src_paths} with {dest}")
	file_list = set()
	for src in src_paths:
		print('-'*80)
		print(f"Getting file list from {src}")
		startTime = time.perf_counter()
		if not parallel_file_listing:
			files,links,init_size,folders = get_file_list_serial(src,exclude=exclude,append_hash=append_hash,full_hash=full_hash)
		else:
			files,links,init_size,folders  = get_file_list_parallel(src, max_workers,exclude=exclude,append_hash=append_hash,full_hash=full_hash)
		if dest_image:
			# we use full path for src when comparing file list with dest image
			file_list.update(trimPaths([os.path.abspath(file) for file in files],'/'))
			file_list.update(trimPaths([os.path.abspath(link) for link in links],'/'))
			file_list.update([folder_path + os.path.sep for folder_path in trimPaths([os.path.abspath(folder) for folder in folders],'/')])
		else:
			file_list.update(trimPaths(files,src))
			file_list.update(trimPaths(links,src))
			file_list.update([folder_path + os.path.sep for folder_path in trimPaths(folders, src)])
		endTime = time.perf_counter()
		print(f"Time taken to get file list: {endTime-startTime:0.4f} seconds")
	startTime = time.perf_counter()
	if not parallel_file_listing:
		files,links,init_size,folders = get_file_list_serial(dest,exclude=exclude,append_hash=append_hash,full_hash=full_hash)
	else:
		files,links,init_size,folders  = get_file_list_parallel(dest, max_workers,exclude=exclude,append_hash=append_hash,full_hash=full_hash)
	file_list2 = set(trimPaths(files,dest))
	file_list2.update(trimPaths(links,dest))
	file_list2.update([folder_path + os.path.sep for folder_path in trimPaths(folders, dest)])
	endTime = time.perf_counter()
	print(f"Time taken to get file list: {endTime-startTime:0.4f} seconds")
	compareFileList(file_list, file_list2, diff_file_list,tar_diff_file_list = tar_diff_file_list)

def createImage(dest_image,target_mount_point,loop_devices: list,src_paths: list, max_workers = 4 * multiprocessing.cpu_count(),
				parallel_file_listing = False,exclude=None):
	if target_mount_point and dest_image:
		# This means we were supplied a dest_image that does not exist, we need to create it and initialize it
		init_size = 0
		for src in src_paths:
			src = os.path.abspath(src + os.path.sep)
			if not parallel_file_listing:
				_,_,size,_ = get_file_list_serial(src,exclude=exclude)
				init_size += size
			else:
				_,_,size,_ = get_file_list_parallel(src, max_workers,exclude=exclude)
				init_size += size
		image_file_size = int(1.05 *init_size + 16*1024*1024) # add 16 MB for the file system
		image_file_size = (int(image_file_size / 4096.0) + 1) * 4096 # round up to the nearest 4 KiB
		print(f"Estimated file size {format_bytes(init_size)}B Creating {dest_image} with size {format_bytes(image_file_size)}B")
		try:
			# use fallocate to allocate the space
			run_command_in_multicmd_with_path_check(["fallocate","-l",str(image_file_size),dest_image])
		except:
			# use python native method to allocate the space
			print("fallocate not available, using python native method to allocate space")
			with open(dest_image, 'wb') as f:
				f.seek(image_file_size-1)
				f.write(b'\0')
		# setup a loop device
		target_loop_device_dest = create_loop_device(dest_image)
		loop_devices.append(target_loop_device_dest)
		# zero the superblocks
		print(f"Clearing {target_loop_device_dest} and create GPT partition table")
		run_command_in_multicmd_with_path_check(['dd','if=/dev/zero','of='+target_loop_device_dest,'bs=1M','count=16'])
		#run_command_in_multicmd_with_path_check(f"parted -s {target_loop_device_dest} mklabel gpt")
		run_command_in_multicmd_with_path_check(['sgdisk','-Z',target_loop_device_dest])

		print(f"Loop device {target_loop_device_dest} created")
		target_partition = get_largest_partition(target_loop_device_dest) # should just return the loop device itself, but just in case.
		# format the partition
		# check if mkudffs is available and image file size is smaller than 8 TiB
		# if shutil.which('mkudffs') and image_file_size < 8 * 1024 * 1024 * 1024 * 1024:
		# 	print(f"Formatting {target_partition} as udf")
		# 	run_command_in_multicmd_with_path_check(f"mkudffs --utf8 --media-type=hd --blocksize=2048 --lvid=HPCP_disk_image --vid=HPCP_img --fsid=HPCP_img --vsid=HPCP_img {target_partition}")
		if shutil.which('mkfs.xfs'):
			print(f"Formatting {target_partition} as xfs")
			run_command_in_multicmd_with_path_check(['mkfs.xfs','-f',target_partition])
		else:
			print(f"Formatting {target_partition} as ext4")
			run_command_in_multicmd_with_path_check(['mkfs.ext4','-F',target_partition])
		# mount the loop device to a temporary folder
		print(f"Mounting {target_partition} at {target_mount_point}")
		run_command_in_multicmd_with_path_check(["mount",target_partition,target_mount_point])
		# verify mount
		if os.path.ismount(target_mount_point):
			return target_mount_point + os.path.sep
		else:
			print(f"Destination image cannot be mounted, exiting.")
			exit(1)
	else:
		print(f"Destination path not specified, exiting.")
		exit(0)

def processCopy(src_paths: list, dest = "", single_thread = False, max_workers = 4 * multiprocessing.cpu_count(),verbose = False, 
				directory_only = False,no_directory_sync = False, full_hash = False, files_per_job = 1, parallel_file_listing = False,
				exclude=None,dest_image = None,batch = False):
	total_file_list = set()
	total_sym_links = {}
	taskCtr = 0
	argDest = dest
	for src in src_paths:
		print('-'*80)
		taskCtr += 1
		dest = argDest
		if dest_image:
			# if the destination is a mounted image, then we use the full path for src and add that to dest.
			src = os.path.abspath(src + os.path.sep)
			# the dest path have the src full src path, we will also recersively copy the dir meta data
			# Gather all parent dirs of src
			srcParentDirs = []
			srcParent = os.path.dirname(src)
			while srcParent != '/':
				srcParentDirs.append(srcParent)
				srcParent = os.path.dirname(srcParent)
			srcParentDirs.reverse()
			# create the parent dirs in dest
			destParentDirs = [ os.path.abspath(dest + srcParentDir + os.path.sep) for srcParentDir in srcParentDirs]
			sync_directory_metadata_bulk(srcParentDirs,destParentDirs)
			dest = os.path.abspath(dest + src + os.path.sep)
		elif os.path.basename(dest) == '' and not os.path.basename(src) == '':
			dest = os.path.join(dest,os.path.basename(src))
			dest += os.path.sep
			src += os.path.sep
			dest = os.path.abspath(dest)
			src = os.path.abspath(src)
		print('-'*80)
		print(f"Task {taskCtr} of {len(src_paths)}, copying from {src} to {dest}")
		# verify dest is writable
		if not os.access(os.path.dirname(os.path.abspath(dest)), os.W_OK):
			print(f"Destination {dest} is not writable, continue with caution.")
			#exit(1)
		if os.path.islink(src):
			total_sym_links[src] = dest
			print(f"{src} is a symlink, creating symlink in dest")
			continue
		if os.path.isfile(src):
			print("Copying single file")
			copy_file(src, dest,full_hash=full_hash,verbose=verbose)
			continue
		if no_directory_sync:
			print("Skipping directory sync")
			sync_directory_metadata(src, dest)
		else:
			startTime = time.perf_counter()
			if single_thread:
				total_sym_links.update(sync_directories_serial(src, dest,exclude=exclude))
			else:
				total_sym_links.update(sync_directories_parallel(src, dest, max_workers,verbose=verbose,exclude=exclude))
			endTime = time.perf_counter()
			print(f"\nTime taken to sync directory: {endTime-startTime:0.4f} seconds")
		if not directory_only:
			global HASH_SIZE
			if HASH_SIZE == 0:
				print("Using file attributes only for skipping")
			elif xxhash_available:
				print("Using xxhash for skipping")
			else:
				print("Using blake2b for skipping")
			if single_thread:
				copy_counter, copy_size_counter , rtnSymLinks , file_list = copy_files_serial(src, dest, full_hash = full_hash,verbose=verbose,exclude=exclude)
			else:
				copy_counter, copy_size_counter , rtnSymLinks , file_list = copy_files_parallel(src, dest, max_workers,full_hash = full_hash,verbose=verbose,files_per_job=files_per_job,parallel_file_listing=parallel_file_listing,exclude=exclude)
			total_file_list.update(trimPaths(file_list,src))
			print(f'Total files copied:     {copy_counter}')
			print(f'Total size copied:      {format_bytes(copy_size_counter)}B')
			print(f'Total files discovered: {len(total_file_list)}')
			total_sym_links.update(rtnSymLinks)
			total_file_list.update(trimPaths(rtnSymLinks.keys(),src))
		print('-'*80)
	return total_file_list, total_sym_links

def verifyDDSrcPath(src_path,loop_devices: list = []):
	dd_src = src_path
	if not dd_src:
		print("DD Source not specified, exiting.")
		exit(1)
	if len(dd_src) != 1:
		print("DD Source is not 1, exiting.")
		exit(1)
	dd_src = dd_src[0]
	if not os.path.exists(dd_src):
		print(f"DD Source {dd_src} does not exist, exiting.")
		exit(1)
	# check if dd_src is a block device
	if not pathlib.Path(dd_src).resolve().is_block_device():
		# check if dd_src is a file
		if not os.path.isfile(dd_src):
			print(f"DD Source {dd_src} is not a block device or a file, exiting.")
			exit(1)
		# mount as a loop device
		print(f"DD Source {dd_src} is a file, mounting as a loop device")
		dd_src = create_loop_device(dd_src,read_only=True)
		loop_devices.append(dd_src)
	return dd_src

def createDDDestPartTable(dd_src,dd_resize = [],src_path = None, dest_path = None):
	src_path = src_path if src_path else dd_src
	if not dest_path:
		print(f"Destination path not specified.")
		return
	partition_infos = get_partition_infos(dd_src)
	disk_name = dd_src
	if len(partition_infos) == 1:
		print(f"Source device {dd_src} is not partitioned, exiting.")
		return
	# sort the partitions by size
	disk_info = partition_infos.pop(disk_name)
	sorted_partitions = sorted(partition_infos.keys(), key=lambda x: partition_infos[x]['size'])
	# change the partitions sizes if provided
	if dd_resize:
		# dd resize is a list of sizes for each partition, we sort it first
		dd_resize = [format_bytes(size,to_int=True) for size in dd_resize]
		dd_resize = sorted(dd_resize,reverse=True)
		# we resize the partitions according to the dd_resize list, 
		# we use as much info from the dd_resize list as possible, assuming it specifies the sizes of the partitions from the largest to the smallest
		# which means if it specified two sizes, we resize the two largest partitions to the specified sizes
		# largest_partition = sorted_partitions[-2]
		# partition_infos[largest_partition]['size'] = format_bytes(dd_resize,to_int=True)
		for i in range(min(len(dd_resize),len(sorted_partitions))):
			partition_infos[sorted_partitions[-i-1]]['size'] = dd_resize[i]

		# recaclulate the disk size, also include 1M extra for each partition 
		disk_info['size'] = sum([partition_infos[partition]['size'] for partition in partition_infos]) + 1024*1024*len(partition_infos)
	partition_infos[disk_name] = disk_info
	sorted_partitions.append(disk_name)
	
	# if dest_path exit, print a confirmation message
	if os.path.exists(dest_path):
		print(f"Warning: Destination path {dest_path} exists.")
		print(f"Source device {src_path} will be copied to {dest_path} with the following partition info:")
		print(partition_infos)
		print(f"All data on {dest_path} will be lost, do you want to continue? (y/n) Default : y")
		inStr = multiCMD.input_with_timeout_and_countdown(60)
		if inStr and not inStr.lower().startswith('y'):
			print(f"Exiting.")
			return
	resize_image(dest_path, partition_infos[disk_name]['size'])
	create_partition_table(dest_path,partition_infos,sorted_partitions)
	print("Image created successfully.")
	return partition_infos

def ddPartition(src_partition_path,dest_partition_path,partition,dd_src,dd_dest):
	# First verify the two partition size is the same
	src_part_info = get_partition_infos(dd_src)
	dest_part_info = get_partition_infos(dd_dest)
	if src_part_info[partition]['size'] > dest_part_info[partition]['size']:
		print(f"Source partition size {src_part_info[partition]['size']} is than the destination partition size {dest_part_info[partition]['size']}.")
		print(f"Cannot use DD, exiting.")
		exit(1)
	run_command_in_multicmd_with_path_check(['dd','if='+src_partition_path,'of='+dest_partition_path,'bs=1024M'])

def cleanUp(mount_points: list,loop_devices: list):
	# clean up loop devices and mount points if we are using a image
	for mount_point in mount_points:
		print(f"Unmounting {mount_point}")
		run_command_in_multicmd_with_path_check(["umount",mount_point])
		print(f"Removing mount point {mount_point}")
		os.rmdir(mount_point)
	for loop_device_dest in loop_devices:
		print(f"Removing loop device {loop_device_dest}")
		run_command_in_multicmd_with_path_check(['losetup','-d',loop_device_dest])

HASH_SIZE = 1<<16

def get_args(args = None):
	parser = argparse.ArgumentParser(description='Copy files from source to destination',
								  epilog=f'Found bins: {list(_binPaths.values())}')
	parser.add_argument('-s', '--single_thread', action='store_true', help='Use serial processing')
	parser.add_argument('-j','-m','-t','--max_workers', type=int, default=4 * multiprocessing.cpu_count(), help='Max workers for parallel processing. Default is 4 * CPU count. Use negative numbers to indicate {n} * CPU count, 0 means 1/2 CPU count.')
	parser.add_argument('-b','--batch',action='store_true', help='Batch mode, process all files in one go')
	parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
	parser.add_argument('-do', '--directory_only', action='store_true', help='Only copy directory structure')
	parser.add_argument('-nds', '--no_directory_sync', action='store_true', help='Do not sync directory metadata, useful for verfication')
	parser.add_argument('-fh', '--full_hash', action='store_true', help='Checks the full hash of files')
	parser.add_argument('-hs', '--hash_size', type=int, default=1<<16, help='Hash size in bytes, default is 65536')
	parser.add_argument('-f', '--files_per_job', type=int, default=1, help='Base number of files per job, will be adjusted dynamically. Default is 1')
	parser.add_argument('-sfl','-lfl', '--source_file_list', type=str, help='Load source file list from file. Will treat it raw meaning do not expand files / folders files are seperated using newline.  If --compare_file_list is specified, it will be used as source for compare')
	parser.add_argument('-fl','-tfl', '--target_file_list', type=str,help='Specify the file_list file to store list of files in src_path to. If --compare_file_list is specified, it will be used as targets for compare')
	parser.add_argument('-cfl', '--compare_file_list',action='store_true', help='Only compare file list. Use --file_list to specify a existing file list or specify the dest_path to compare src_path with. When not using with file_list, will compare hash.')
	parser.add_argument('-dfl', '--diff_file_list', type=str, nargs='?', const="auto",default=None, help="Implies --compare_file_list, specify a file name to store the diff file list to or omit the value to auto-determine.")
	parser.add_argument('-tdfl', '--tar_diff_file_list', action='store_true', help='Generate a tar compatible diff file list. ( update / new files only )')
	parser.add_argument('-nhfl', '--no_hash_file_list', action='store_true', help='Do not append hash to file list')
	parser.add_argument('-rm', '--remove', action='store_true', help='Remove all files and folders specified in src_path')
	parser.add_argument('-rf', '--remove_force', action='store_true', help='Remove all files without prompt')
	parser.add_argument('-rme', '--remove_extra', action='store_true', help='Remove all files and folders in dest_path that are not in src_path')
	parser.add_argument('-e', '--exclude', action='append', default=[], help='Exclude source files matching the pattern')
	parser.add_argument('-x', '--exclude_file', type=str, help='Exclude source files matching the pattern in the file')
	parser.add_argument('-nlt', '--no_link_tracking', action='store_true', help='Do not copy files that symlinks point to.')
	parser.add_argument('-V', '--version', action='version', version=f"%(prog)s {version} with {('XXHash' if xxhash_available else 'Blake2b')} and multiCMD V{multiCMD.version}; High Performance CoPy (HPC coPy) by pan@zopyr.us")
	parser.add_argument('-pfl', '--parallel_file_listing', action='store_true', help='Use parallel processing for file listing')
	parser.add_argument('src_path', nargs='*', type=str, help='Source Path')
	parser.add_argument('-si','--src_image', nargs='*', type=str, help='Source Image, mount the image and copy the files from it.')
	parser.add_argument('-siff','--load_diff_image', nargs='*', type=str, help='Not implemented. Load diff images and apply the changes to the destination.')
	parser.add_argument('-d','-C','--dest_path', type=str, help='Destination Path')
	parser.add_argument('-di','--dest_image', type=str, help='Destination Image, create a image file and copy the files into it.')
	parser.add_argument('-dis','--dest_image_size', type=str, help='Not implemented. Destination Image Size, specify the size of the destination image to split into. Default is 0 (No split).', default='0')
	parser.add_argument('-diff', '--get_diff_image', action='store_true', help='Not implemented. Compare the source and destination file list, create a diff image of that will update the destination to source.')
	parser.add_argument('-dd', '--disk_dump', action='store_true', help='Disk to Disk mirror, use this if you are backuping / deploying an OS from / to a disk. \
					 Require 1 source, can be 1 src_path or 1 -si src_image, require 1 -di dest_image.')
	parser.add_argument('-ddr', '--dd_resize', action='append', type=str, help='Resize the destination image to the specified size with dd')
	#args = parser.parse_args(args)
	try:
		args = parser.parse_intermixed_args(args)
	except Exception as e:
		#eprint(f"Error while parsing arguments: {e!r}")
		# try to parse the arguments using parse_known_args
		args, unknown = parser.parse_known_args()
		# if there are unknown arguments, we will try to parse them again using parse_args
		if unknown:
			print(f"Warning: Unknown arguments, treating all as Source Path: {unknown!r}")
			args.src_path = args.src_path + unknown
	#print(f'Arguments: {vars(args)}')
	defualt_args_dict = vars(parser.parse_args([]))
	# format a long format argument of what the user supplied and echo it back
	startArgs = [f'> {sys.argv[0]}']
	for values in args.src_path:
		startArgs.append(f'\'{values}\'')
	for argumentName, value in vars(args).items():
		if value != defualt_args_dict[argumentName]:
			if argumentName == 'src_path':
				continue
			if isinstance(value, list):
				# skip positional arguments

				for v in value:
					startArgs.append(f'--{argumentName}=\'{v}\'')
			else:
				startArgs.append(f'--{argumentName}=\'{value}\'')
	print(' '.join(startArgs))
	return args

# ---- Main Function ----
def hpcp(src_path, dest_path = "", single_thread = False, max_workers = 4 * multiprocessing.cpu_count(),
			verbose = False, directory_only = False,no_directory_sync = False, full_hash = False, files_per_job = 1, target_file_list = "",
			compare_file_list = False, diff_file_list = None, tar_diff_file_list = False, remove = False,remove_force = False, remove_extra = False, parallel_file_listing = False,
			exclude=None,exclude_file = None,dest_image = None,dest_image_size = '0', no_link_tracking = False,src_image = None,dd = False,dd_resize = 0,
			batch = False, append_hash_to_file_list = True, hash_size = ..., source_file_list = None):
	global HASH_SIZE
	if hash_size != ...:
		
		try:
			HASH_SIZE = int(hash_size)
		except:
			print(f"Invalid hash size {hash_size}, using default hash size {HASH_SIZE}")
	if HASH_SIZE < 0:
		HASH_SIZE = 0
	if HASH_SIZE == 0:
		print("Warning: Hash size set to 0, will not check file content for skipping.")
	print('-'*80)
	src_paths = []
	src_images = []
	mount_points = []
	loop_devices = []
	src_str = ''
	dest_str = ''
	programStartTime = time.perf_counter()
	exclude = formatExclude(exclude,exclude_file)
	if max_workers == 0:
		max_workers = round(0.5 * multiprocessing.cpu_count())
	elif max_workers < 0:
		max_workers = round(- max_workers * multiprocessing.cpu_count())

	if dd:
		if os.name == 'nt':
			print("dd mode is not supported on Windows, exiting")
			return(0)
		print("dd mode enabled, performing Disk Dump Copy. Setting up the target ...")

		dest_path = dest_path if dest_path else dest_image
		src_path = src_path if src_path else src_image

		if not dest_path:
			if src_path:
				print(f"Destination path not specified, using {src_path[-1]} as destination")
				dest_path = src_path.pop()

		# check write permission on dest_path
		if not os.access(os.path.dirname(os.path.abspath(dest_path)), os.W_OK):
			print(f"Destination path {dest_path} is not writable, continuing with high probability of failure.")
			#exit(1)
		dd_src = verifyDDSrcPath(src_path,loop_devices = loop_devices)
		partition_infos = createDDDestPartTable(dd_src,dd_resize=dd_resize,src_path=src_path, dest_path=dest_path)
		if not partition_infos:
			return 1
		
		dd_dest = dest_path
		# check if dd_dest is a block device
		if not pathlib.Path(dd_dest).resolve().is_block_device():
			# check if dd_dest is a file
			if os.path.isfile(dd_dest):
				print(f"DD Destination {dd_dest} is a file, mounting as a loop device")
				dd_dest = create_loop_device(dd_dest)
				loop_devices.append(dd_dest)
		
		disk_info = partition_infos.pop(dd_src)
		# need to check if partion info is empty and fix partition table if necessary
		src_partition_paths = get_partitions(dd_src)
		dest_partition_paths = get_partitions(dd_dest)
		for partition in partition_infos:
			print(f"Copying partition {partition} from {dd_src} to {dd_dest}")
			# mount both the src and dest to a temporary folder
			src_mount_point = tempfile.mkdtemp()
			mount_points.append(src_mount_point)
			dest_mount_point = tempfile.mkdtemp()
			mount_points.append(dest_mount_point)
			src_partition_path = [path for path in src_partition_paths if path.endswith(partition)][0]
			dest_partition_path = [path for path in dest_partition_paths if path.endswith(partition)][0]
			print(f"Mounting {src_partition_path} at {src_mount_point}")
			run_command_in_multicmd_with_path_check(['mount',src_partition_path,src_mount_point])
			# check if the mount is successful
			if not any(os.scandir(src_mount_point)) and not os.path.ismount(src_mount_point):
				print(f"Error mounting {src_partition_path}, usig dd for copying.")
				ddPartition(src_partition_path,dest_partition_path,partition,dd_src,dd_dest)
				continue
			print(f"Mounting {dest_partition_path} at {dest_mount_point}")
			run_command_in_multicmd_with_path_check(['mount',dest_partition_path,dest_mount_point])
			# check if the mount is successful
			if not any(os.scandir(dest_mount_point)) and not os.path.ismount(dest_mount_point):
				print(f"Error mounting {dest_partition_path}, usig dd for copying.")
				ddPartition(src_partition_path,dest_partition_path,partition,dd_src,dd_dest)
				continue
			# copy the partition files
			print(f"Copying partition {partition} files from {src_mount_point} to {dest_mount_point}")
			hpcp([src_mount_point], dest_path = dest_mount_point, single_thread=single_thread, max_workers=max_workers,
																verbose=verbose, directory_only=directory_only,no_directory_sync=no_directory_sync,
																full_hash=full_hash, files_per_job=files_per_job, parallel_file_listing=parallel_file_listing,
																exclude=exclude,no_link_tracking = True)
		cleanUp(mount_points,loop_devices)
		# sort the output partitions
		#run_command_in_multicmd_with_path_check(f"sgdisk --sort {dest_path}")
		print(f"Done disk dumping {src_path} to {dest_path}.")
		return 0
		
	# set max_workers to 61 on windows
	if os.name == 'nt':
		if dest_image or src_image:
			print("Destination / Source as a image is currently not supported on Windows, exiting")
			return(0)
		max_workers = min(max_workers,61)
		if max_workers == 61:
			print(f"Max workers set to 61 on Windows")
			print("This is because Windows has a limit of 64 threads per process and we need 3 threads for the main process")
			print("See https://bugs.python.org/issue26903")

	# if not dest_path:
	#     dest_path = src_path.pop()
	if src_image:
		src_str = mountSrcImage(src_image,src_images,src_paths,mount_points,loop_devices)

	if source_file_list:
		src_paths.extend(loadFileList(source_file_list))

	verifySrcPath(src_path,src_paths)
	if not src_str:
		src_str = "-".join([os.path.basename(src) for src in src_paths])

	if target_file_list:
		storeFileList(target_file_list,src_paths,single_thread=single_thread, max_workers=max_workers,verbose=verbose, 
						files_per_job=files_per_job,compare_file_list=compare_file_list, remove_extra=remove_extra,
						parallel_file_listing=parallel_file_listing,exclude=exclude,diff_file_list=diff_file_list,tar_diff_file_list=tar_diff_file_list,src_str = src_str,
						append_hash=append_hash_to_file_list,full_hash=full_hash)
		cleanUp(mount_points,loop_devices)
		return 0


	if dest_image:
		dest , target_mount_point = getDestFromImage(dest_image,mount_points,loop_devices)
		dest_str = dest_image
	else:
		dest = getDestFromPath(dest_path,src_paths,src_path,can_be_none=remove)
		dest_str = dest


	if compare_file_list or diff_file_list:
		if diff_file_list == 'auto':
			if not src_str:
				src_str = '-'.join([os.path.basename(os.path.realpath(src)) for src in src_paths])
			diff_file_list = f'DIFF_{src_str}_TO_{os.path.basename(os.path.realpath(dest_str))}_{int(time.time())}_{"tar_" if tar_diff_file_list else ""}file_list.txt'
		processCompareFileList(src_paths, dest, max_workers=max_workers, file_list=target_file_list,parallel_file_listing=parallel_file_listing,
						 exclude=exclude,dest_image=dest_image,diff_file_list=diff_file_list,tar_diff_file_list=tar_diff_file_list,append_hash=append_hash_to_file_list,full_hash=full_hash)
		cleanUp(mount_points,loop_devices)
		return 0
	
	try:
		if dest and dest != ... and dest.endswith(os.path.sep) and (not (os.path.exists(dest) or os.path.ismount(dest))):
			os.makedirs(dest, exist_ok=True)
	except FileExistsError as e:
		print(f"Destination path {dest} maybe a mounted dir, known issue with os.path.exists\nContinuing without creating dest folder...")

	if not dest:
		if remove:
			dest = ...
			dest_str = dest_path
		else:
			dest = createImage(dest_image,target_mount_point,loop_devices,src_paths,max_workers=max_workers,parallel_file_listing=parallel_file_listing,exclude=exclude)

	#TODO: support dest_image_size
	if dest != ...:
		total_file_list, total_sym_links = processCopy(src_paths, dest, single_thread=single_thread, max_workers=max_workers,
																		verbose=verbose, directory_only=directory_only,no_directory_sync=no_directory_sync,
																		full_hash=full_hash, files_per_job=files_per_job, parallel_file_listing=parallel_file_listing,
																		exclude=exclude,dest_image=dest_image,batch = batch)
		if verbose:
			# sort file list and sym links
			for file in natural_sort(total_file_list):
				print(f"Copied File: {file}")
			for link in total_sym_links:
				print(f"Link: {link} -> {total_sym_links[link]}")
		creatSymLinks(total_sym_links,exclude=exclude,no_link_tracking=no_link_tracking)
		if remove_extra:
			print('-'*80)
			remove_extra_files(total_file_list, dest,max_workers,verbose,files_per_job,single_thread,exclude=exclude)
			print('-'*80)
			print("Removing extra empty directories...")
			remove_extra_dirs(src_paths, dest,exclude=exclude)
		print('-'*80)
		if len(src_paths) > 1:
			print("Overall Summary:")
			print(f"Number of files / links: {len(total_file_list)}")
			print(f"Number of links: {len(total_sym_links)}")
			print(f"Total time taken: {time.perf_counter()-programStartTime:0.4f} seconds")
	if remove:
		processRemove(src_paths,single_thread=single_thread, max_workers=max_workers,verbose=verbose,
					  files_per_job=files_per_job, remove_force=remove_force,exclude=exclude,batch=batch)
	cleanUp(mount_points,loop_devices)
	print(f"Done.")
	# we exit if we are not involved with a gui
		
def hpcp_gui():
	import tkinter as tk
	from tkinter import filedialog, messagebox
	global root
	root = tk.Tk()
	root.title("High Performance CoPy")

	src_entry = tk.Entry(root)
	src_entry.grid(row=0, column=1)
	tk.Label(root, text="Source path").grid(row=0)

	src_browse_button = tk.Button(root, text="Browse", command=lambda: src_entry.insert(0, filedialog.askdirectory()))
	src_browse_button.grid(row=0, column=2)

	dest_entry = tk.Entry(root)
	dest_entry.grid(row=1, column=1)
	tk.Label(root, text="Destination path").grid(row=1)

	dest_browse_button = tk.Button(root, text="Browse", command=lambda: dest_entry.insert(0, filedialog.askdirectory()))
	dest_browse_button.grid(row=1, column=2)

	var_max_workers = tk.IntVar(value=4 * multiprocessing.cpu_count())
	tk.Entry(root, textvariable=var_max_workers).grid(row=2, column=1)
	tk.Label(root, text="Max Workers").grid(row=2)

	var_single_thread = tk.BooleanVar()
	tk.Checkbutton(root, text="Single Thread", variable=var_single_thread).grid(row=3, column=0)

	var_parallel_file_listing = tk.BooleanVar()
	tk.Checkbutton(root, text="Parallel File Listing", variable=var_parallel_file_listing).grid(row=3, column=1)

	var_directory_only = tk.BooleanVar()
	tk.Checkbutton(root, text="Directory Only", variable=var_directory_only).grid(row=4, column=0)

	var_full_hash = tk.BooleanVar()
	tk.Checkbutton(root, text="Full Hash", variable=var_full_hash).grid(row=4, column=1)

	var_compare_file_list = tk.BooleanVar()
	tk.Checkbutton(root, text="Compare File List", variable=var_compare_file_list).grid(row=5, column=0)

	var_verbose = tk.BooleanVar()
	tk.Checkbutton(root, text="Verbose", variable=var_verbose).grid(row=5, column=1)

	var_files_per_job = tk.IntVar(value=1)
	tk.Entry(root, textvariable=var_files_per_job).grid(row=6, column=1)
	tk.Label(root, text="Files per Job").grid(row=6)

	var_file_list = tk.StringVar()
	# file list
	tk.Label(root, text="Save / Load File List .txt",).grid(row=7, column=0, sticky='w')
	tk.Entry(root, textvariable=var_file_list).grid(row=7, column=1)
	tk.Button(root, text="Browse", command=lambda: var_file_list.set(filedialog.askopenfilename(filetypes=(("Text Files", "*.txt"),("All Files", "*.*"))))).grid(row=7, column=2)



	var_remove = tk.BooleanVar()
	tk.Checkbutton(root, text="Delete Mode", variable=var_remove).grid(row=8, column=0)

	var_remove_extra = tk.BooleanVar()
	tk.Checkbutton(root, text="Delete Extra In Dest", variable=var_remove_extra).grid(row=8, column=1)





	def run_hpcp():
		try:
			root.withdraw()
			print(f"Source path: {src_entry.get()}")
			print(f"Destination path: {dest_entry.get()}")
			# we print out any arguments that are not default
			if var_single_thread.get():
				print(f"Single Thread: {var_single_thread.get()}")
			if var_max_workers.get() != 4 * multiprocessing.cpu_count():
				print(f"Max Workers: {var_max_workers.get()}")
			if var_verbose.get():
				print(f"Verbose: {var_verbose.get()}")
			if var_directory_only.get():
				print(f"Directory Only: {var_directory_only.get()}")
			if var_full_hash.get():
				print(f"Full Hash: {var_full_hash.get()}")
			if var_files_per_job.get() != 1:
				print(f"Files per Job: {var_files_per_job.get()}")
			if var_file_list.get():
				print(f"File List .txt: {var_file_list.get()}")
			if var_compare_file_list.get():
				print(f"Compare File List: {var_compare_file_list.get()}")
			if var_remove.get():
				print(f"Remove: {var_remove.get()}")
			if var_remove_extra.get():
				print(f"Remove Extra: {var_remove_extra.get()}")
			if var_parallel_file_listing.get(): 
				print(f"Parallel File Listing: {var_parallel_file_listing.get()}")

			hpcp([src_entry.get()], dest_entry.get(), single_thread=var_single_thread.get(), max_workers=var_max_workers.get(),
				verbose=var_verbose.get(), directory_only=var_directory_only.get(), full_hash=var_full_hash.get(), files_per_job=var_files_per_job.get(),
				target_file_list=var_file_list.get(), compare_file_list=var_compare_file_list.get(), remove=var_remove.get(), remove_extra=var_remove_extra.get(),
				parallel_file_listing=var_parallel_file_listing.get())
			# messagebox.showinfo("Success", "File copy completed successfully")
			root.deiconify()
		except Exception as e:
			messagebox.showerror("Error", str(e))

	tk.Button(root, text="Start Copy", command=run_hpcp).grid(row=13)
	root.mainloop()

# ---- CLI ----
def main():
	args = get_args()
	# we run gui if the current platform is windows and src_path is not specified
	if os.name == 'nt' and len(args.src_path) == 0:
		hpcp_gui()
	else:
		rtnCode = hpcp(args.src_path, dest_path = args.dest_path, single_thread = args.single_thread, max_workers = args.max_workers, verbose = args.verbose,
			 directory_only =  args.directory_only, no_directory_sync = args.no_directory_sync,full_hash = args.full_hash, files_per_job = args.files_per_job,
			 target_file_list = args.target_file_list, compare_file_list = args.compare_file_list , diff_file_list = args.diff_file_list, tar_diff_file_list = args.tar_diff_file_list,remove = args.remove, remove_force =args.remove_force,
			 remove_extra = args.remove_extra, parallel_file_listing = args.parallel_file_listing,exclude = args.exclude,exclude_file = args.exclude_file,
			 dest_image = args.dest_image,dest_image_size=args.dest_image_size,no_link_tracking = args.no_link_tracking,src_image = args.src_image,dd=args.disk_dump,
			 dd_resize=args.dd_resize,batch=args.batch,append_hash_to_file_list=not args.no_hash_file_list, hash_size=args.hash_size,source_file_list=args.source_file_list)
		if rtnCode:
			exit(rtnCode)
		

if __name__ == '__main__':
	main()