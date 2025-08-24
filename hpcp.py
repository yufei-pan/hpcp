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
import stat
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
from math import log
try:
	import multiCMD
	assert float(multiCMD.version) > 1.19
except:
	import time,threading,io,argparse,sys,subprocess,select,os,string,re,itertools,signal
	class multiCMD:
		version='1.33_min'
		__version__=version
		__running_threads=set()
		__variables={}
		class Task:
			def __init__(A,command):A.command=command;A.returncode=None;A.stdout=[];A.stderr=[];A.thread=None;A.stop=False
			def __iter__(A):return zip(['command','returncode','stdout','stderr'],[A.command,A.returncode,A.stdout,A.stderr])
			def __repr__(A):return f"Task(command={A.command}, returncode={A.returncode}, stdout={A.stdout}, stderr={A.stderr}, stop={A.stop})"
			def __str__(A):return str(dict(A))
			def is_alive(A):
				if A.thread is not None:return A.thread.is_alive()
				return False
		class AsyncExecutor:
			def __init__(A,max_threads=1,semaphore=...,timeout=0,quiet=True,dry_run=False,parse=False):
				C=max_threads;B=semaphore;A.max_threads=C
				if B is...:B=threading.Semaphore(C)
				A.semaphore=B;A.runningThreads=[];A.tasks=[];A.timeout=timeout;A.quiet=quiet;A.dry_run=dry_run;A.parse=parse;A.__lastNotJoined=0
			def __iter__(A):return iter(A.tasks)
			def __repr__(A):return f"AsyncExecutor(max_threads={A.max_threads}, semaphore={A.semaphore}, runningThreads={A.runningThreads}, tasks={A.tasks}, timeout={A.timeout}, quiet={A.quiet}, dry_run={A.dry_run}, parse={A.parse})"
			def __str__(A):return str(A.tasks)
			def __len__(A):return len(A.tasks)
			def __bool__(A):return bool(A.tasks)
			def run_commands(A,commands,timeout=...,max_threads=...,quiet=...,dry_run=...,parse=...,sem=...):
				G=sem;F=parse;E=dry_run;D=quiet;C=max_threads;B=timeout
				if B is...:B=A.timeout
				if C is...:C=A.max_threads
				if D is...:D=A.quiet
				if E is...:E=A.dry_run
				if F is...:F=A.parse
				if G is...:G=A.semaphore
				if len(A.runningThreads)>130000:
					A.wait(timeout=0)
					if len(A.runningThreads)>130000:
						print('The amount of running threads approching cpython limit of 130704. Waiting until some available.')
						while len(A.runningThreads)>120000:A.wait(timeout=1)
				elif len(A.runningThreads)+A.__lastNotJoined>1000:A.wait(timeout=0);A.__lastNotJoined=len(A.runningThreads)
				H=multiCMD.run_commands(commands,timeout=B,max_threads=C,quiet=D,dry_run=E,with_stdErr=False,return_code_only=False,return_object=True,parse=F,wait_for_return=False,sem=G);A.tasks.extend(H);A.runningThreads.extend([A.thread for A in H]);return H
			def run_command(A,command,timeout=...,max_threads=...,quiet=...,dry_run=...,parse=...,sem=...):return A.run_commands([command],timeout=timeout,max_threads=max_threads,quiet=quiet,dry_run=dry_run,parse=parse,sem=sem)[0]
			def wait(A,timeout=...,threads=...):
				C=threads;B=timeout
				if C is...:C=A.runningThreads
				if B is...:B=A.timeout
				for D in C:
					if B>=0:D.join(timeout=B)
					else:D.join()
				A.runningThreads=[A for A in A.runningThreads if A.is_alive()];return A.runningThreads
			def stop(A,timeout=...):
				for B in A.tasks:B.stop=True
				A.wait(timeout);return A.tasks
			def cleanup(A,timeout=...):A.stop(timeout);A.tasks=[];A.runningThreads=[];return A.tasks
			def join(B,timeout=...,threads=...,print_error=True):
				B.wait(timeout=timeout,threads=threads)
				for A in B.tasks:
					if A.returncode!=0 and print_error:print(f"Command: {A.command} failed with return code: {A.returncode}");print('Stdout:');print('\n  '.join(A.stdout));print('Stderr:');print('\n  '.join(A.stderr))
				return B.tasks
			def get_results(A,with_stdErr=False):
				if with_stdErr:return[A.stdout+A.stderr for A in A.tasks]
				else:return[A.stdout for A in A.tasks]
			def get_return_codes(A):return[A.returncode for A in A.tasks]
		_BRACKET_RX=re.compile('\\[([^\\]]+)\\]')
		_ALPHANUM=string.digits+string.ascii_letters
		_ALPHA_IDX={B:A for(A,B)in enumerate(_ALPHANUM)}
		def _expand_piece(piece,vars_):
			D=vars_;C=piece;C=C.strip()
			if':'in C:E,F,G=C.partition(':');D[E]=G;return
			if'-'in C:
				A,F,B=(A.strip()for A in C.partition('-'));A=D.get(A,A);B=D.get(B,B)
				if A.isdigit()and B.isdigit():H=max(len(A),len(B));return[f"{A:0{H}d}"for A in range(int(A),int(B)+1)]
				if all(A in string.hexdigits for A in A+B):return[format(A,'x')for A in range(int(A,16),int(B,16)+1)]
				try:return[multiCMD._ALPHANUM[A]for A in range(multiCMD._ALPHA_IDX[A],multiCMD._ALPHA_IDX[B]+1)]
				except KeyError:pass
			return[D.get(C,C)]
		def _expand_ranges_fast(inStr):
			D=inStr;global __variables;A=[];B=0
			for C in multiCMD._BRACKET_RX.finditer(D):
				if C.start()>B:A.append([D[B:C.start()]])
				E=[]
				for G in C.group(1).split(','):
					F=multiCMD._expand_piece(G,__variables)
					if F:E.extend(F)
				A.append(E or['']);B=C.end()
			A.append([D[B:]]);return[''.join(A)for A in itertools.product(*A)]
		def _expand_ranges(inStr):
			global __variables;E=[inStr];G=[];I=string.digits+string.ascii_letters
			while len(E)>0:
				C=E.pop();F=re.search('\\[(.*?)]',C)
				if not F:G.append(C);continue
				J=F.group(1);K=J.split(',')
				for D in K:
					D=D.strip()
					if':'in D:L,M,D=D.partition(':');__variables[L]=D;E.append(C.replace(F.group(0),'',1))
					elif'-'in D:
						try:A,M,B=D.partition('-')
						except ValueError:G.append(C);continue
						A=A.strip()
						if A in __variables:A=__variables[A]
						B=B.strip()
						if B in __variables:B=__variables[B]
						if A.isdigit()and B.isdigit():
							N=min(len(A),len(B));O='{:0'+str(N)+'d}'
							for H in range(int(A),int(B)+1):P=O.format(H);E.append(C.replace(F.group(0),P,1))
						elif all(A in string.hexdigits for A in A+B):
							for H in range(int(A,16),int(B,16)+1):E.append(C.replace(F.group(0),format(H,'x'),1))
						else:
							try:
								Q=I.index(A);R=I.index(B)
								for H in range(Q,R+1):E.append(C.replace(F.group(0),I[H],1))
							except ValueError:G.append(C)
					else:E.append(C.replace(F.group(0),D,1))
			G.reverse();return G
		def __handle_stream(stream,target,pre='',post='',quiet=False):
			E=quiet;C=target
			def D(current_line,target,keepLastLine=True):
				A=target
				if not keepLastLine:
					if not E:sys.stdout.write('\r')
					A.pop()
				elif not E:sys.stdout.write('\n')
				B=current_line.decode('utf-8',errors='backslashreplace');A.append(B)
				if not E:sys.stdout.write(pre+B+post);sys.stdout.flush()
			A=bytearray();B=True
			for F in iter(lambda:stream.read(1),b''):
				if F==b'\n':
					if not B and A:D(A,C,keepLastLine=False)
					elif B:D(A,C,keepLastLine=True)
					A=bytearray();B=True
				elif F==b'\r':D(A,C,keepLastLine=B);A=bytearray();B=False
				else:A.extend(F)
			if A:D(A,C,keepLastLine=B)
		def int_to_color(n,brightness_threshold=500):
			B=brightness_threshold;A=hash(str(n));C=A>>16&255;D=A>>8&255;E=A&255
			if C+D+E<B:return multiCMD.int_to_color(A,B)
			return C,D,E
		def __run_command(task,sem,timeout=60,quiet=False,dry_run=False,with_stdErr=False,identity=None):
			I=timeout;F=identity;E=quiet;A=task;C='';D=''
			with sem:
				try:
					if F is not None:
						if F==...:F=threading.get_ident()
						P,Q,R=multiCMD.int_to_color(F);C=f"[38;2;{P};{Q};{R}m";D='\x1b[0m'
					if not E:print(C+'Running command: '+' '.join(A.command)+D);print(C+'-'*100+D)
					if dry_run:return A.stdout+A.stderr
					B=subprocess.Popen(A.command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,stdin=subprocess.PIPE);J=threading.Thread(target=multiCMD.__handle_stream,args=(B.stdout,A.stdout,C,D,E),daemon=True);J.start();K=threading.Thread(target=multiCMD.__handle_stream,args=(B.stderr,A.stderr,C,D,E),daemon=True);K.start();L=time.time();M=len(A.stdout)+len(A.stderr);time.sleep(0);H=1e-07
					while B.poll()is None:
						if A.stop:B.send_signal(signal.SIGINT);time.sleep(.01);B.terminate();break
						if I>0:
							if len(A.stdout)+len(A.stderr)!=M:L=time.time();M=len(A.stdout)+len(A.stderr)
							elif time.time()-L>I:A.stderr.append('Timeout!');B.send_signal(signal.SIGINT);time.sleep(.01);B.terminate();break
						time.sleep(H)
						if H<.001:H*=2
					A.returncode=B.poll();J.join(timeout=1);K.join(timeout=1);N,O=B.communicate()
					if N:multiCMD.__handle_stream(io.BytesIO(N),A.stdout,A)
					if O:multiCMD.__handle_stream(io.BytesIO(O),A.stderr,A)
					if A.returncode is None:
						if A.stderr and A.stderr[-1].strip().startswith('Timeout!'):A.returncode=124
						elif A.stderr and A.stderr[-1].strip().startswith('Ctrl C detected, Emergency Stop!'):A.returncode=137
						else:A.returncode=-1
				except FileNotFoundError as G:print(f"Command / path not found: {A.command[0]}",file=sys.stderr,flush=True);A.stderr.append(str(G));A.returncode=127
				except Exception as G:import traceback as S;print(f"Error running command: {A.command}",file=sys.stderr,flush=True);print(str(G).split('\n'));A.stderr.extend(str(G).split('\n'));A.stderr.extend(S.format_exc().split('\n'));A.returncode=-1
				if not E:print(C+'\n'+'-'*100+D);print(C+f"Process exited with return code {A.returncode}"+D)
				if with_stdErr:return A.stdout+A.stderr
				else:return A.stdout
		def ping(hosts,timeout=1,max_threads=0,quiet=True,dry_run=False,with_stdErr=False,return_code_only=False,return_object=False,wait_for_return=True,return_true_false=True):
			E=return_true_false;D=return_code_only;B=hosts;C=False
			if isinstance(B,str):F=[f"ping -c 1 {B}"];C=True
			else:F=[f"ping -c 1 {A}"for A in B]
			if E:D=True
			A=multiCMD.run_commands(F,timeout=timeout,max_threads=max_threads,quiet=quiet,dry_run=dry_run,with_stdErr=with_stdErr,return_code_only=D,return_object=return_object,wait_for_return=wait_for_return)
			if E:
				if C:return not A[0]
				else:return[not A for A in A]
			elif C:return A[0]
			else:return A
		def run_command(command,timeout=0,max_threads=1,quiet=False,dry_run=False,with_stdErr=False,return_code_only=False,return_object=False,wait_for_return=True,sem=None):return multiCMD.run_commands(commands=[command],timeout=timeout,max_threads=max_threads,quiet=quiet,dry_run=dry_run,with_stdErr=with_stdErr,return_code_only=return_code_only,return_object=return_object,parse=False,wait_for_return=wait_for_return,sem=sem)[0]
		def __format_command(command,expand=False):
			D=expand;A=command
			if isinstance(A,str):
				if D:B=multiCMD._expand_ranges_fast(A)
				else:B=[A]
				return[A.split()for A in B]
			elif hasattr(A,'__iter__'):
				C=[]
				for E in A:
					if isinstance(E,str):C.append(E)
					else:C.append(repr(E))
				if not D:return[C]
				F=[multiCMD._expand_ranges_fast(A)for A in C];B=list(itertools.product(*F));return[list(A)for A in B]
			else:return multiCMD.__format_command(str(A),expand=D)
		def run_commands(commands,timeout=0,max_threads=1,quiet=False,dry_run=False,with_stdErr=False,return_code_only=False,return_object=False,parse=False,wait_for_return=True,sem=None):
			K=wait_for_return;J=dry_run;I=quiet;H=timeout;C=max_threads;B=sem;E=[]
			for L in commands:E.extend(multiCMD.__format_command(L,expand=parse))
			A=[multiCMD.Task(A)for A in E]
			if C<1:C=len(E)
			if C>1 or not K:
				if not B:B=threading.Semaphore(C)
				F=[threading.Thread(target=multiCMD.__run_command,args=(A,B,H,I,J,...),daemon=True)for A in A]
				for(D,G)in zip(F,A):G.thread=D;D.start()
				if K:
					for D in F:D.join()
				else:__running_threads.update(F)
			else:
				B=threading.Semaphore(1)
				for G in A:multiCMD.__run_command(G,B,H,I,J,identity=None)
			if return_code_only:return[A.returncode for A in A]
			elif return_object:return A
			elif with_stdErr:return[A.stdout+A.stderr for A in A]
			else:return[A.stdout for A in A]
		def join_threads(threads=__running_threads,timeout=None):
			A=threads;global __running_threads
			for B in A:B.join(timeout=timeout)
			if A is __running_threads:__running_threads={A for A in A if A.is_alive()}
		def input_with_timeout_and_countdown(timeout,prompt='Please enter your selection'):
			B=prompt;A=timeout;print(f"{B} [{A}s]: ",end='',flush=True)
			for C in range(A,0,-1):
				if sys.stdin in select.select([sys.stdin],[],[],0)[0]:return input().strip()
				print(f"\r{B} [{C}s]: ",end='',flush=True);time.sleep(1)
		def _genrate_progress_bar(iteration,total,prefix='',suffix='',columns=120):
			G=columns;F=prefix;E=total;C=suffix;B=iteration;J=False;K=False;L=False;M=False
			if E==0:return f"{F} iteration:{B} {C}".ljust(G)
			N=f"|{'{0:.1f}'.format(100*(B/float(E)))}% ";A=G-len(F)-len(C)-len(N)-3
			if A<=0:A=G-len(F)-len(C)-3;L=True
			if A<=0:A=G-len(C)-3;J=True
			if A<=0:A=G-3;K=True
			if A<=0:return f"""{F}
iteration:
{B}
total:
{E}
| {C}
"""
			if B==0:M=True
			H=int(A*B//E);I='▁▂▃▄▅▆▇█';P=A*B/E-H;Q=int(P*(len(I)-1));R=I[Q]
			if H==A:O=I[-1]*A
			else:O=I[-1]*H+R+'_'*(A-H)
			D=''
			if not J:D+=F
			if not M:
				D+=f"{O}"
				if not L:D+=N
			elif A>=16:D+=f" Calculating... "
			if not K:D+=C
			return D
		def get_terminal_size():
			try:import os;A=os.get_terminal_size()
			except:
				try:import fcntl,termios as C,struct as B;D=fcntl.ioctl(0,C.TIOCGWINSZ,B.pack('HHHH',0,0,0,0));A=B.unpack('HHHH',D)[:2]
				except:import shutil as E;A=E.get_terminal_size(fallback=(120,30))
			return A
		def print_progress_bar(iteration,total,prefix='',suffix=''):
			D=prefix;C=total;B=iteration;A=suffix;D+=' |'if not D.endswith(' |')else'';A=f"| {A}"if not A.startswith('| ')else A
			try:
				E,F=multiCMD.get_terminal_size();sys.stdout.write(f"\r{multiCMD._genrate_progress_bar(B,C,D,A,E)}");sys.stdout.flush()
				if B==C and C>0:print(file=sys.stdout)
			except:
				if B%5==0:print(multiCMD._genrate_progress_bar(B,C,D,A))
		def main(self):A=argparse.ArgumentParser(description='Run multiple commands in parallel');A.add_argument('commands',metavar='command',type=str,nargs='+',help='commands to run');A.add_argument('-p','--parse',action='store_true',help='Parse ranged input and expand them into multiple commands');A.add_argument('-t','--timeout',metavar='timeout',type=int,default=60,help='timeout for each command');A.add_argument('-m','--max_threads',metavar='max_threads',type=int,default=1,help='maximum number of threads to use');A.add_argument('-q','--quiet',action='store_true',help='quiet mode');A.add_argument('-V','--version',action='version',version=f"%(prog)s {self.version} by pan@zopyr.us");B=A.parse_args();multiCMD.run_commands(B.commands,B.timeout,B.max_threads,B.quiet,parse=B.parse,with_stdErr=True)

try:
	import xxhash
	hasher = xxhash.xxh64()
	xxhash_available = True
except ImportError:
	import hashlib
	hasher = hashlib.blake2b()
	xxhash_available = False

version = '9.28'
__version__ = version
COMMIT_DATE = '2025-08-23'

MAGIC_NUMBER = 1.61803398875
RANDOM_DESTINATION_SELECTION = False

BYTES_RATE_LIMIT = 0
FILES_RATE_LIMIT = 0

#%% ---- Helper Functions ----
class Adaptive_Progress_Bar:
	def __init__(self, total_count = 0, total_size = 0,refresh_interval = 0.1,last_num_job_for_stats = 10,custom_prefix = None,
			  custom_suffix = None,process_word = 'Processed',use_print_thread = False, suppress_all_output = False,bytes_rate_limit = 0,files_rate_limit = 0):
		self.total_count = total_count
		self.total_size = total_size
		self.refresh_interval = refresh_interval
		self.item_counter = 0
		self.size_counter = 0
		self.scheduled_jobs = 0
		self.stop_flag = suppress_all_output
		self.process_word = process_word
		self.start_time = time.perf_counter()
		self.last_n_jobs = deque(maxlen=last_num_job_for_stats)
		self.custom_prefix = custom_prefix
		self.custom_suffix = custom_suffix
		self.last_call_args = None
		self.use_print_thread = use_print_thread
		self.quiet = suppress_all_output
		self.bytes_rate_limit = bytes_rate_limit
		self.files_rate_limit = files_rate_limit
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
		if self.quiet:
			return
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
			total_time = time.perf_counter() - self.start_time
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
			#if self.scheduled_jobs:
			prefix += f' {self.scheduled_jobs} Scheduled'
			if job_count > 0:
				prefix += f' {files_per_job:0>3.1f} F/Job '
		if self.custom_suffix:
			suffix = self.custom_suffix
		else:
			suffix = f'{format_bytes(total_size_speed)}B/s {format_bytes(total_file_speed,use_1024_bytes=False,to_str=True)}F/s |'
			if self.bytes_rate_limit > 0 or self.files_rate_limit > 0:
				if self.bytes_rate_limit > 0:
					suffix += f' {format_bytes(self.bytes_rate_limit)}B/s Limit |'
				if self.files_rate_limit > 0:
					suffix += f' {format_bytes(self.files_rate_limit,use_1024_bytes=False,to_str=True)}F/s Limit |'
			elif job_count > 0:
				suffix += f' {last_n_time:.1f}s: {format_bytes(last_n_size_speed)}B/s {format_bytes(last_n_file_speed,use_1024_bytes=False,to_str=True)}F/s |'
			suffix += f' {format_time(remaining_time)}'
		callArgs = (self.item_counter, self.total_count, prefix, suffix)
		if callArgs != self.last_call_args:
			self.last_call_args = callArgs
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
	def under_rate_limit(self):
		# calculate the sleep time based on the rate limits
		if self.total_count == self.item_counter:
			return True
		if not self.bytes_rate_limit and not self.files_rate_limit:
			return True
		duration = time.perf_counter() - self.start_time
		if self.bytes_rate_limit > 0:
			estimated_max_current_copy_size = self.bytes_rate_limit * duration
			if self.size_counter > estimated_max_current_copy_size:
				return False
		if self.files_rate_limit > 0:
			estimated_max_current_copy_files = self.files_rate_limit * duration
			if self.item_counter > estimated_max_current_copy_files:
				return False
		return True
	def rate_limit(self):
		while not self.under_rate_limit():
			# sleep for 0.1 seconds
			self.print_progress()
			time.sleep(self.refresh_interval)
			


_binPaths = {}
@functools.lru_cache(maxsize=None)
def check_path(program_name):
	"""
	Check if the given program is in the system path.

	Args:
		program_name (str): The name of the program to check.

	Returns:
		bool: True if the program is found, False otherwise.
	"""
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

_binCalled = {'lsblk', 'losetup', 'sgdisk', 'blkid', 'umount', 'mount','dd','cp', 'xcopy',
			  'truncate', 
			  'mkfs', 'mkfs.btrfs', 'mkfs.xfs', 'mkfs.ntfs', 'mkfs.vfat', 'mkfs.exfat', 'mkfs.hfsplus', 
			  'mkudffs', 'mkfs.jfs', 'mkfs.reiserfs', 'newfs', 'mkfs.bfs', 'mkfs.minix', 'mkswap',
			  'e2fsck', 'btrfs', 'xfs_repair', 'ntfsfix', 'fsck.fat', 'fsck.exfat', 'fsck.hfsplus', 
			  'fsck.hfs', 'fsck.jfs', 'fsck.reiserfs', 'fsck.ufs', 'fsck.minix'}
[check_path(program) for program in _binCalled]

def run_command_in_multicmd_with_path_check(command, timeout=0,max_threads=1,quiet=False,dry_run=False,strict=False):
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
	if command[0] not in _binPaths and not check_path(command[0]):
		print(f"Error: Command '{command[0]}' not found. Please consider installing it then retry.", file=sys.stderr, flush=True)
		if strict: 
			sys.exit(127)
	# Run the command
	task = multiCMD.run_commands([command], timeout=timeout, max_threads=max_threads, quiet=quiet, dry_run=dry_run,return_object=True)[0]
	if task.returncode != 0:
		if not quiet:
			print(f"Error: Command '{command}' failed with return code {task.returncode}.", file=sys.stderr, flush=True)
		if strict:
			raise RuntimeError(f"Command '{command}' failed with return code {task.returncode}.")
	return task.stdout

def get_free_space_bytes(path):
	stat = os.statvfs(path)
	return stat.f_bavail * stat.f_frsize  # available blocks * fragment size

def get_file_size(path):
	try:
		st = os.stat(path,follow_symlinks=False)
		if 'st_rsize' in st:
			realSize = st.st_rsize
		elif 'st_blocks' in st:
			realSize = st.st_blocks * 512
		else:
			realSize = st.st_size
	except:
		try:
			realSize = os.path.getsize(path)
		except:
			realSize = 0
	return realSize

#%% -- Exclude --
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

def format_exclude(exclude = None,exclude_file = None) -> frozenset:
	"""
	Format and normalize exclusion patterns for path matching.

	This function processes exclusion patterns from both a list and an optional file.
	It normalizes paths by replacing consecutive slashes with a single slash and
	ensures patterns have appropriate prefixes for glob-style matching.

	Parameters:
		exclude (iterable, optional): Collection of path patterns to exclude. 
			Defaults to None (empty set).
		exclude_file (str, optional): Path to a file containing exclusion patterns, 
			one per line. Defaults to None.

	Returns:
		frozenset: A frozen set of normalized exclusion patterns.

	Note:
		- Patterns not starting with '/' will have '*/' prepended unless they already start with '*/'.
		- The function handles errors gracefully if the exclude_file doesn't exist or can't be read.
	"""
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

#%% -- DD --
def get_largest_partition(disk):
	"""
	Get the largest partition on the disk.

	Args:
		disk (str): The disk name or path.

	Returns:
		str: The path of the largest partition on the disk.
	"""
	partitions = run_command_in_multicmd_with_path_check(["lsblk", '-nbl', '-o', 'NAME,SIZE',disk],strict=True)
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
	partitions = run_command_in_multicmd_with_path_check(["lsblk", '-nbl', '-o', 'NAME,SIZE',disk],strict=True)
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
	fs_type = run_command_in_multicmd_with_path_check(["lsblk", '-nbl', '-o', 'FSTYPE', path],strict=True)[0].strip()
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
	"""
	Create a loop device for the image file.

	Args:
		image_path (str): The path to the image file.
		read_only (bool, optional): Whether to create the loop device in read-only mode. Defaults to False.

	Returns:
		str: The path to the created loop device.
	"""
	if read_only:
		loop_device_dest = run_command_in_multicmd_with_path_check(["losetup", '--partscan', '--find', '--show', '--read-only', image_path],strict=True)[0].strip()
	else:
		loop_device_dest = run_command_in_multicmd_with_path_check(["losetup", '--partscan', '--find', '--show', image_path],strict=True)[0].strip()
	#run_command_in_multicmd_with_path_check(f'partprobe {loop_device_dest}')
	print(f"Loop device {loop_device_dest} created.")
	return loop_device_dest

def get_target_partition(image, partition_name):
	"""
	Gets the device path for a specific partition within an image file or block device.
	
	This function handles both regular image files and block devices. If the input is an
	image file, it creates a loop device first. It then identifies all partitions and
	returns the path to the partition that matches the given partition name.
	
	Args:
		image (str): Path to the disk image file or block device
		partition_name (str): Name or suffix of the target partition to find
		
	Returns:
		tuple: A tuple containing:
			- str: Path to the target partition (e.g., '/dev/loop0p1')
			- str or None: Path to the loop device if one was created, otherwise None
			
	Note:
		If the image is not a block device, a loop device will be created using
		the create_loop_device function and will need to be cleaned up by the caller.
	"""
	loop_device = None
	if not pathlib.Path(image).resolve().is_block_device():
		loop_device = create_loop_device(image)
		image = loop_device
	# Need to get a partition path for mkfs
	partitions = run_command_in_multicmd_with_path_check(["lsblk", '-nbl', '-o', 'NAME', image],strict=True)
	partitions.pop(0) # remove the disk itself
	target_partition = ''
	for part in partitions:
		if part.endswith(partition_name):
			target_partition = '/dev/' + part
			break
	return target_partition, loop_device

@functools.lru_cache(maxsize=None)
def get_partition_details(device, partition,sector_size=512):
	"""
	Retrieves detailed information about a specific partition on a device.

	This function gathers partition information including GUID codes, partition names,
	attributes, filesystem type, UUID, label, and size using tools like sgdisk and blkid.

	Args:
		device (str): Path to the device containing the partition (e.g., '/dev/sda').
		partition (str): The partition number or identifier (e.g., '1' for first partition).
		sector_size (int, optional): Sector size in bytes. Defaults to 512.

	Returns:
		dict: A dictionary containing partition details with the following keys:
			- 'partition_guid_code': The GUID code identifying the partition type.
			- 'unique_partition_guid': The unique GUID identifying this specific partition.
			- 'partition_name': The name of the partition if available.
			- 'partition_attrs': Attribute flags for the partition.
			- 'fs_type': The filesystem type (e.g., 'ext4', 'ntfs').
			- 'fs_uuid': The UUID of the filesystem.
			- 'fs_label': The filesystem label if available.
			- 'size': Size of the partition in bytes.

	Notes:
		- This function uses external tools (sgdisk, blkid) and requires appropriate permissions.
		- If the device is an image file, a loop device is temporarily created and then detached.
	"""
	# Get the partition info from the source
	result = run_command_in_multicmd_with_path_check(["sgdisk", '--info='+partition, device],strict=True)
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
	"""
	Extracts detailed partition information from a specified block device.
	
	This function parses the output of the 'sgdisk --print' command to gather 
	information about partitions on the device, including disk size, identifier, 
	sector size, and individual partition details.
	
	Args:
		device (str): Path to the block device (e.g., '/dev/sda').
		
	Returns:
		dict: A dictionary containing disk and partition information with the following structure:
			{
				'disk_name': {
					'size': int,             # Total disk size in bytes
					'disk_identifier': str,   # Disk UUID/identifier
					'sector_size': int        # Sector size in bytes
				},
				'partition_number': {         # Partition details from get_partition_details()
					...
				},
				...
			}
	
	Note:
		This function depends on 'run_command_in_multicmd_with_path_check' and 'get_partition_details'.
	"""
	# partitions = run_command_in_multicmd_with_path_check(f"lsblk -nbl -o NAME,SIZE {device}")
	# partition_info = {part.split()[0]: int(part.split()[1]) for part in partitions}
	partitions = run_command_in_multicmd_with_path_check(["sgdisk", '--print', device],strict=True)
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
	Writes partition information to a specified partition within a disk image.
	
	This function handles the configuration of partition attributes, GUID codes,
	unique GUIDs, and file system creation. It supports various file systems including
	ext2/3/4, btrfs, xfs, ntfs, fat/vfat variants, exfat, hfs/hfsplus, udf, jfs,
	reiserfs, and others.
	
	Args:
		image (str): Path to the disk image file to be modified.
		partition_infos (dict): Dictionary containing partition configuration details.
			Expected keys for the specified partition_name:
			- 'partition_guid_code': GUID type code for the partition.
			- 'unique_partition_guid': Unique GUID for the partition.
			- 'partition_attrs': Hexadecimal string of partition attributes.
			- 'fs_type': File system type to create.
			- 'fs_label': Label for the file system.
			- 'fs_uuid': UUID for the file system.
			- 'partition_name': Name for the partition.
		partition_name (str): The partition identifier to be configured.
	
	Raises:
		Exception: If any error occurs during the partition information writing process.
	
	Note:
		Some file systems have limitations on setting labels or UUIDs, and appropriate
		warnings will be printed in these cases. Special handling is applied for read-only
		file systems like cramfs and iso9660.
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
				command = ['mkfs.btrfs']
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
				print("Skip creating zfs file system. ZFS file system should be created using zpool command.")
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
		run_command_in_multicmd_with_path_check(["truncate", f'--size={format_bytes(total_size,to_int=True)}', image],strict=True)

def is_device(path):
    mode = os.stat(path).st_mode
    return stat.S_ISCHR(mode) or stat.S_ISBLK(mode)

def get_mount_table():
	"""
	Get the mount table of the system.

	Returns:
		dict: A dictionary where the keys are device paths and the values are lists of mount points.
	"""
	mount_table = {}
	with open('/proc/mounts', 'r') as f:
		for line in f:
			parts = line.split()
			if len(parts) >= 2:
				mount_table.setdefault(parts[0], []).append(parts[1])
	return mount_table

#%% -- Symbolic Links --
def create_sym_links(symLinks,exclude=None,no_link_tracking=False):
	global RANDOM_DESTINATION_SELECTION
	if len(symLinks) == 0:
		return
	nestedSymLinks = {}
	counter = 0
	print(f"\nFound Symbolic Links:   {len(symLinks)}")
	if no_link_tracking:
		print(f"Skipping copying file as no_link_tracking ...\n")
	#print(symLinks)
	start_time = time.perf_counter()
	for src, dests in symLinks.items():
		try:
			src = os.path.normpath(src)
			dests = [os.path.normpath(d) for d in dests]
			if exclude and is_excluded(src,exclude):
				print(f"\n{src} is excluded, skipping...")
				continue
			dest = ''
			for d in dests:
				if os.path.islink(d):
					os.unlink(d)
				dest = d
				if os.path.exists(d):
					if os.path.isdir(d):
						print(f"\n{d} is a directory, skipping...")
						#shutil.rmtree(dest)
						continue
					else:
						print(f"\n{d} is a file, skipping...")
						#os.remove(dest)
						continue
			# Determine if the link is a absolute link or relative link
			linkedTargetFile = os.readlink(src)
			if not os.path.isabs(linkedTargetFile) and not no_link_tracking:
				sourceLinkedFile = os.path.join(os.path.dirname(src), linkedTargetFile)
				# we also copy the pointed file if the file doesn't exist
				destLinkedFiles = [os.path.join(os.path.dirname(d), linkedTargetFile) for d in dests]
				for d,destLinkedFile in zip(dests,destLinkedFiles):
					if os.path.exists(destLinkedFile):
						dest = d
						break
				if not dest:
					if not os.path.exists(sourceLinkedFile):
						print(f"\nFile {sourceLinkedFile} which is linked by {src} doesn't exist! \nSkipping copying original file...")
					else:
						_, _ , rtnSymLinks , _ = copy_files_serial(sourceLinkedFile, destLinkedFiles,exclude=exclude)
						nestedSymLinks.update(rtnSymLinks)
			while dests:
				if not dest:
					if RANDOM_DESTINATION_SELECTION:
						idx = random.randrange(len(dests))
					else:
						idx = 0
					dest = dests.pop(idx)
				elif dest in dests:
					dests.remove(dest)
				try:
					os.symlink(linkedTargetFile, dest, target_is_directory=os.path.isdir(linkedTargetFile))
					break
				except:
					print(f'Could not create symbolic link from {linkedTargetFile} to {dest}')
					if dests:
						print(f"Trying next destination...")
						dest = ''
						continue
					else:
						print(f"All destinations failed, skipping...")
						break
			counter += 1
			# print the progress bar with the total count and the speed in F/s
			prefix = f'{counter} Symbolic Links Created'
			suffix = f'{counter / (time.perf_counter() - start_time):.2f} F/s'
			multiCMD.print_progress_bar(counter, len(symLinks), prefix=prefix, suffix=suffix)
			# we catch the file name too long exception
		except OSError as e:
			print("Exception caught! Possibly file name too long!")
			print(f"\n{e}")
			print(f"\n{src} -> {dests}")
			print("Skipping...")
			continue
		except Exception as e:
			print("Exception caught!")
			print(f"\n{e}")
			print(f"\n{src} -> {dests}")
			print("Skipping...")
			continue


	endTime = time.perf_counter()
	print(f"Time taken:             {endTime-start_time:0.4f} seconds")
	if len(nestedSymLinks) > 0:
		print(f"\nNested Symbolic Links:   {len(nestedSymLinks)}")
		create_sym_links(nestedSymLinks,exclude=exclude,no_link_tracking=no_link_tracking)

#%% -- File list --
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

#%% -- Format --
def format_bytes(size, use_1024_bytes=None, to_int=False, to_str=False,str_format='.2f'):
	"""
	Format the size in bytes to a human-readable format or vice versa.
	From hpcp: https://github.com/yufei-pan/hpcp

	Args:
		size (int or str): The size in bytes or a string representation of the size.
		use_1024_bytes (bool, optional): Whether to use 1024 bytes as the base for conversion. If None, it will be determined automatically. Default is None.
		to_int (bool, optional): Whether to convert the size to an integer. Default is False.
		to_str (bool, optional): Whether to convert the size to a string representation. Default is False.
		str_format (str, optional): The format string to use when converting the size to a string. Default is '.2f'.

	Returns:
		int or str: The formatted size based on the provided arguments.

	Examples:
		>>> format_bytes(1500, use_1024_bytes=False)
		'1.50 K'
		>>> format_bytes('1.5 GiB', to_int=True)
		1610612736
		>>> format_bytes('1.5 GiB', to_str=True)
		'1.50 Gi'
		>>> format_bytes(1610612736, use_1024_bytes=True, to_str=True)
		'1.50 Gi'
		>>> format_bytes(1610612736, use_1024_bytes=False, to_str=True)
		'1.61 G'
	"""
	if to_int or isinstance(size, str):
		if isinstance(size, int):
			return size
		elif isinstance(size, str):
			# Use regular expression to split the numeric part from the unit, handling optional whitespace
			match = re.match(r"(\d+(\.\d+)?)\s*([a-zA-Z]*)", size)
			if not match:
				if to_str:
					return size
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
			unit_labels = {'': 0, 'k': 1, 'm': 2, 'g': 3, 't': 4, 'p': 5, 'e': 6, 'z': 7, 'y': 8}
			if unit not in unit_labels:
				if to_str:
					return size
				print(f"Invalid unit '{unit}'. Expected one of {list(unit_labels.keys())}")
				return 0
			if to_str:
				return format_bytes(size=int(number * (power ** unit_labels[unit])), use_1024_bytes=use_1024_bytes, to_str=True, str_format=str_format)
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
				size = size.rstrip('B').rstrip('b')
				size = float(size.lower().strip())
			except Exception as e:
				return size
		# size is in bytes
		if use_1024_bytes or use_1024_bytes is None:
			power = 2**10
			n = 0
			power_labels = {0 : '', 1: 'Ki', 2: 'Mi', 3: 'Gi', 4: 'Ti', 5: 'Pi', 6: 'Ei', 7: 'Zi', 8: 'Yi'}
			while size > power:
				size /= power
				n += 1
			return f"{size:{str_format}}{' '}{power_labels[n]}"
		else:
			power = 10**3
			n = 0
			power_labels = {0 : '', 1: 'K', 2: 'M', 3: 'G', 4: 'T', 5: 'P', 6: 'E', 7: 'Z', 8: 'Y'}
			while size > power:
				size /= power
				n += 1
			return f"{size:{str_format}}{' '}{power_labels[n]}"
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

#%% -- Hash --
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

#%% -- Path --
def trim_paths(paths, baseDir):
	"""
	Convert a set of absolute paths to relative paths based on a base directory.
	
	Args:
		paths (set or list): A collection of absolute file paths.
		baseDir (str): The base directory to make paths relative to.
	
	Returns:
		set: A set of file paths, each relative to the parent directory of baseDir.
	
	Example:
		>>> trim_paths({'/home/user/project/file1.py', '/home/user/project/file2.py'}, '/home/user/project/main.py')
		{'file1.py', 'file2.py'}
	"""
	return set([os.path.relpath(path,os.path.dirname(baseDir)) for path in paths])

#%% ---- Generate File List ----
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
		realSize = get_file_size(root)
		return frozenset([get_file_repr(root,append_hash,full_hash)]) ,frozenset(), realSize,frozenset()
	file_list = set()
	links = set()
	folders = set()
	size = 0
	iteration = 0
	start_time = time.perf_counter()
	globalStartTIme = start_time
	if os.path.isdir(root):
		folders.add(root)
		for entry in os.scandir(root):
			# update the progress bar every 0.5 seconds
			currentTime = time.perf_counter()
			if currentTime - start_time > 0.5:
				start_time = currentTime
				# use the time passed as the iteration number
				iteration = int(currentTime - globalStartTIme)
				# if the root is longer than 50 characters, we only show the last 50 characters
				multiCMD.print_progress_bar(iteration=iteration, total=0, prefix=f'{root}'[-50:], suffix=f'Files: {format_bytes(len(file_list),use_1024_bytes=False,to_str=True)} Links: {format_bytes(len(links),use_1024_bytes=False,to_str=True)} Folders: {format_bytes(len(folders),use_1024_bytes=False,to_str=True)} Size: {format_bytes(size)}B')
			if exclude and is_excluded(entry.path,exclude):
				continue
			if entry.is_symlink():
				links.add(get_file_repr(entry.path,append_hash,full_hash))
				realSize = get_file_size(entry.path)
				size += realSize
			elif entry.is_file(follow_symlinks=False):
				file_list.add(get_file_repr(entry.path,append_hash,full_hash))
				realSize = get_file_size(entry.path)
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
		realSize = get_file_size(path)
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

#%% ---- Delete Files ----
def delete_file_bulk(paths):
	total_size = 0
	start_time = time.perf_counter()
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
	return total_size, endTime - start_time

def delete_file_list_parallel(file_list, max_workers, verbose=False,files_per_job=1,init_size=0):
	global FILES_RATE_LIMIT
	global BYTES_RATE_LIMIT
	total_files = len(file_list)
	file_list_iterator = iter(file_list)
	start_time = time.perf_counter()
	last_refresh_time = start_time
	futures = {}
	if FILES_RATE_LIMIT or BYTES_RATE_LIMIT:
		max_scheduled_jobs = max_workers
	else:
		max_scheduled_jobs = max_workers * 1.2
	files_per_job = max(1,files_per_job)
	apb = Adaptive_Progress_Bar(total_count=total_files,total_size=init_size,last_num_job_for_stats=max(1,max_workers // 2),process_word='Deleted',
							 use_print_thread = True,suppress_all_output=verbose,bytes_rate_limit=BYTES_RATE_LIMIT,files_rate_limit=FILES_RATE_LIMIT)
	with ProcessPoolExecutor(max_workers=max_workers) as executor:
		while file_list_iterator or futures:
			# counter = 0
			while file_list_iterator and len(futures) < max_scheduled_jobs and last_refresh_time - time.perf_counter() < 5 and apb.under_rate_limit():
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

			done, _ = concurrent.futures.wait(futures.keys(), return_when=concurrent.futures.FIRST_COMPLETED, timeout=1)

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
			if done:
				if verbose:
					print(f'\nAverage rmtime is {current_iteration_total_run_time / len(done):0.2f} for {len(done)} jobs with {deleted_files_count_this_run} files each')
				if file_list_iterator and deleted_files_count_this_run == files_per_job and (current_iteration_total_run_time / len(done) > 5 or time.perf_counter() - last_refresh_time > 5):
					files_per_job //= MAGIC_NUMBER
					files_per_job = round(files_per_job)
					if verbose:
						print(f'\nCompletion time is long, changing files per job to {files_per_job}')
				elif file_list_iterator and deleted_files_count_this_run == files_per_job and current_iteration_total_run_time / len(done) < 1:
					files_per_job *= MAGIC_NUMBER
					files_per_job = round(files_per_job)
					if verbose:
						print(f'\nCompletion time is short, changing files per job to {files_per_job}')
				if files_per_job < 1:
					files_per_job = 1
			else:
				if not apb.under_rate_limit():
					if verbose:
						print(f'\nWe had hit the rate limit, changing files per job to 1')
					files_per_job = 1
				time.sleep(apb.refresh_interval)
				apb.print_progress()
			last_refresh_time = time.perf_counter()
	endTime = time.perf_counter()
	apb.stop()
	print(f"\nTime taken:             {endTime-start_time:0.4f} seconds")
	print(f"Average speed:          {format_bytes(apb.size_counter / (endTime-start_time))}B/s")
	print(f"                        {apb.item_counter / (endTime-start_time):.2f} file/s")
	return apb.item_counter, apb.size_counter

def delete_files_parallel(paths, max_workers, verbose=False,files_per_job=1,exclude=None,batch=False):
	if not batch and isinstance(paths, str):
		paths = [paths]
	start_time = time.perf_counter()
	all_files = set()
	init_size_all = 0
	for path in paths:
		file_list, links,init_size, _ = get_file_list_serial(path,exclude=exclude)
		#file_list, links,init_size, folders = get_file_list_parallel(path, max_workers,exclude=exclude)
		all_files.update(set(file_list) | set(links))
		init_size_all += init_size
	endTime = time.perf_counter()
	print(f"Time taken to get file list: {endTime-start_time:0.4f} seconds")
	total_files = len(all_files)
	print(f"Number of files: {total_files}")
	print(f'Initial estimated size: {format_bytes(init_size_all)}B')
	if total_files == 0:
		return 1 , delete_file_bulk(paths)[0]
	delete_counter, delete_size_counter = delete_file_list_parallel(all_files, max_workers, verbose,files_per_job,init_size=init_size)
	print("Removing directory structures....")
	delete_size_counter += delete_file_bulk(paths)[0]
	print(f"Initial estimated size: {format_bytes(init_size_all)}B, Final size: {format_bytes(delete_size_counter)}B")
	return delete_counter + 1, delete_size_counter

#%% ---- Copy Files ----
def copy_file(src_path, dest_paths, full_hash=False, verbose=False, concurrent_processes=0):
	"""
	Copy a file from the source path to the destination path.

	Args:
		src_path (str): The path of the source file.
		dest_path (list): The list of paths of the destination file.
		full_hash (bool, optional): Whether to perform a full hash comparison to determine if the files are identical. Defaults to False.
		verbose (bool, optional): Whether to print verbose output. Defaults to False.
		concurrent_processes (int, optional): The number of concurrent processes running in parralle. Used to calculate preemptive back-off to next dest threashold. Defaults to 0 ( Do not back-off ).

	Returns:
		tuple: A tuple containing the size of the copied file, the time taken for the copy operation, and a dictionary of symbolic links encountered during the copy.
	"""
	global RANDOM_DESTINATION_SELECTION
	symLinks = {}
	#task_to_run = []
	# skip if src path or dest path is longer than 4096 characters
	if len(src_path) > 4096:
		print(f'\nSkipped {src_path} because path is too long')
		return 0, 0, symLinks #, task_to_run
	newDests = []
	inDests = dest_paths
	for dest in dest_paths:
		if len(dest) > 4096:
			print(f'\nSkipped {dest} because path is too long')
		else:
			newDests.append(dest)
	dest_paths = newDests
	if len(dest_paths) == 0:
		print(f'\nSkipped {src_path} because all destination paths are too long')
		return 0, 0, symLinks
	start_time = time.perf_counter()
	try:
		src_size = get_file_size(src_path)
		copiedSize = 0
		for dest in dest_paths:
			if os.path.exists(dest) and (not os.path.islink(src_path)) and is_file_identical(src_path, dest,src_size,full_hash):
				# if verbose:
				#     print(f'\nSkipped {src_path}')
				st = os.stat(src_path,follow_symlinks=False)
				shutil.copystat(src_path, dest,follow_symlinks=False)
				if os.name == 'posix':
					os.chown(dest, st.st_uid, st.st_gid)
					#also copy the modes
					os.chmod(dest, st.st_mode)
				os.utime(dest, (st.st_atime, st.st_mtime),follow_symlinks=False)
				endTime = time.perf_counter()
				return 0, endTime - start_time , symLinks #, task_to_run
		bak_dest_paths = dest_paths.copy()
		using_bak_paths = False
		if os.path.islink(src_path):
			symLinks[src_path] = dest_paths
		else:
			copied = False
			while (not copied) and dest_paths:
				if RANDOM_DESTINATION_SELECTION:
					idx = random.randrange(len(dest_paths))
				else:
					idx = 0
				dest_path = dest_paths.pop(idx)
				dest_free_space = get_free_space_bytes(os.path.dirname(dest_path))
				if not using_bak_paths:
					to_skip = False
					if dest_free_space < src_size:
						if verbose:
							print(f'\nEstimation: not enough space on {dest_path} to copy {src_path}')
							print(f'Free space: {format_bytes(dest_free_space)}B, Required: {format_bytes(src_size)}B')
						to_skip = True
					if not to_skip and concurrent_processes > 0:
						estimated_concurrent_write_size = max(src_size * log(concurrent_processes),1)
						backoff_threashold = dest_free_space / estimated_concurrent_write_size
						if backoff_threashold < random.random():
							if verbose:
								print(f'\nPreemptively backing off on {dest_path} to copy {src_path} with chance {1- backoff_threashold:.2f}')
								print(f'Estimated concurrent write size: {format_bytes(estimated_concurrent_write_size)}B, Free space: {format_bytes(dest_free_space)}B')
							to_skip = True
						elif verbose:
							print(f'\nLottery Win. Continuing on {dest_path} to copy {src_path} with chance {backoff_threashold:.2f}')
							print(f'Estimated concurrent write size: {format_bytes(estimated_concurrent_write_size)}B, Free space: {format_bytes(dest_free_space)}B')
					if to_skip:
						if not dest_paths:
							using_bak_paths = True
							dest_paths = bak_dest_paths
							if verbose:
								print(f'Pre Gracious copy pass failed. Force trying all destination paths for {src_path}')
						continue
				if verbose:
					print(f'\nTrying to copy from {src_path} to {dest_path}')
				try:
					try:
						# if the parent path for the file does not exist, create it
						os.makedirs(os.path.dirname(dest_path), exist_ok=True)
						if os.name == 'posix':
							run_command_in_multicmd_with_path_check(["cp", "-af", "--sparse=always", src_path, dest_path],timeout=0,quiet=True,strict=True)
							copiedSize = get_file_size(dest_path)
						else:
							shutil.copy2(src_path, dest_path, follow_symlinks=False)
							#shutil.copystat(src_path, dest_path)
						copied = True
						break
					except Exception as e:
						if os.path.exists(dest_path):
							os.remove(dest_path)
						if using_bak_paths:
							if not dest_paths and not verbose:
								import traceback
								print(f'Error copying {src_path} to {dest_path}: {e}')
								print(traceback.format_exc())
								print(f'Trying to copy from {src_path} to {dest_path} without sparse')
							if os.name == 'posix':
								run_command_in_multicmd_with_path_check(["cp", "-af", src_path, dest_path],timeout=0,quiet=True,strict=True)
								#task_to_run = ["cp", "-af", src_path, dest_path]
							elif os.name == 'nt':
								run_command_in_multicmd_with_path_check(["xcopy", "/I", "/E", "/Y", "/c", "/q", "/k", "/r", "/h", "/x", src_path, dest_path],timeout=0,quiet=True,strict=True)
								#task_to_run = ["xcopy", "/I", "/E", "/Y", "/c", "/q", "/k", "/r", "/h", "/x", src_path, dest_path]
						elif verbose:
							print(f'Retrying with a different destination path in {dest_paths}')
				except Exception as e:
					if using_bak_paths and not dest_paths:
						import traceback
						print(f'ERROR copying {src_path} to {dest_path}: {e}')
						print(traceback.format_exc())
						print(f'No more destination paths to try')
						return 0, time.perf_counter() - start_time, symLinks #, task_to_run
					elif verbose:
						print(f'Re-Retrying with a different destination path in {dest_paths}')
					if os.path.exists(dest_path):
						os.remove(dest_path)
				if not copied and not using_bak_paths and not dest_paths:
					if verbose:
						print(f'Gracious copy pass failed. Force trying all destination paths for {src_path}')
					using_bak_paths = True
					dest_paths = bak_dest_paths
			if not copied:
				print(f'ERROR: FAILED to copy {src_path} to {dest_paths}')
				return 0, time.perf_counter() - start_time, symLinks
			elif verbose:
				print(f'Copied {src_path} to {dest_path}')
				print(f'Estimated remaining size: {format_bytes(dest_free_space - copiedSize)}B')
	except Exception as e:
		print(f'Fatal Error copying {src_path} to {inDests}: {e}')
		import traceback
		print(traceback.format_exc())
		return 0, time.perf_counter() - start_time, symLinks #, task_to_run
	if not copiedSize:
		copiedSize = src_size
	endTime = time.perf_counter()
	return copiedSize, endTime - start_time , symLinks #, task_to_run

def copy_files_bulk(src_files, dst_files,src_path, full_hash=False, verbose=False, concurrent_processes=0):
	"""
	Copy multiple files from source to destination.

	Args:
		src_files (list): List of source file paths.
		dst_files (list): List of original destination file paths.
		src_path (str): Source directory path.
		full_hash (bool, optional): Whether to calculate full hash of files. Defaults to False.
		verbose (bool, optional): Whether to display verbose output. Defaults to False.
		concurrent_processes (int, optional): Number of concurrent processes to pass to copy_file. Defaults to 0.

	Returns:
		tuple: A tuple containing the total size of copied files, total time taken for copying, and a dictionary of symbolic links.

	"""
	total_size = 0
	total_time = 0
	symLinks = {}
	# tasks_to_run = []
	for src in src_files:
		source_relative_path = os.path.relpath(src, src_path)
		dests = [os.path.join(dest_path, source_relative_path) for dest_path in dst_files]
		size , cpTime , rtnSymLinks = copy_file(src, dests, full_hash, verbose,concurrent_processes)
		total_size += size
		total_time += cpTime
		symLinks.update(rtnSymLinks)
	return total_size , total_time, symLinks

def copy_files_bulk_batch(current_jobs, full_hash=False, verbose=False, concurrent_processes=0):
	total_size = 0
	total_time = 0
	symLinks = {}
	# tasks_to_run = []
	for (src_path, dest_paths, f) in current_jobs:
		source_relative_path = os.path.relpath(f, src_path)
		if source_relative_path == '.':
			dests = dest_paths
		else:
			dests = [os.path.join(dest_path, source_relative_path) for dest_path in dest_paths]
		size , cpTime , rtnSymLinks = copy_file(f, dests, full_hash, verbose,concurrent_processes)
		total_size += size
		total_time += cpTime
		symLinks.update(rtnSymLinks)
	return total_size , total_time, symLinks

def copy_file_list_parallel(file_list, links, src_path, dest_paths, max_workers, full_hash=False, verbose=False, files_per_job=1,estimated_size = 0):
	"""
	Copy a list of files in parallel using multiple workers.

	Args:
		file_list (list): List of file paths to be copied.
		links (list): List of symbolic links to be created.
		src_path (str): Source directory path.
		dest_paths (list): Destination directory path lists.
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
	global FILES_RATE_LIMIT
	global BYTES_RATE_LIMIT
	if FILES_RATE_LIMIT or BYTES_RATE_LIMIT:
		max_scheduled_jobs = max_workers
	else:
		max_scheduled_jobs = max_workers * 1.1
	if len(src_path) > 4096:
		print(f'\nSkipped {src_path} because path is too long')
		return 0, 0, {}, frozenset()
	newDests = []
	for dest in dest_paths:
		if len(dest) > 4096:
			print(f'\nSkipped {dest} because path is too long')
		else:
			newDests.append(dest)
	dest_paths = newDests
	if not dest_paths:
		print(f'\nSkipped {src_path} because all destination paths are too long')
		return 0, 0, {}, frozenset()
	file_list_iterator = iter(file_list)
	start_time = time.perf_counter()
	lastRefreshTime = start_time
	total_files = len(file_list)
	futures = {}
	files_per_job = max(1,files_per_job)
	symLinks = {}
	for link in links:
		srcRelativePath = os.path.relpath(link, src_path)
		symLinks[link] = [os.path.join(dest_path,srcRelativePath) for dest_path in dest_paths]
	if len(file_list) == 0:
		return 0, 0, symLinks , frozenset()
	print(f"Processing {len(file_list)} files with {max_workers} workers")
	apb = Adaptive_Progress_Bar(total_count=total_files,total_size=estimated_size,last_num_job_for_stats=max(1,max_workers//10),process_word='Copied',use_print_thread = True,suppress_all_output=verbose,bytes_rate_limit=BYTES_RATE_LIMIT,files_rate_limit=FILES_RATE_LIMIT)
	with ProcessPoolExecutor(max_workers=max_workers) as executor:
		while file_list_iterator or futures:
			# counter = 0
			while file_list_iterator and len(futures) < max_scheduled_jobs and time.perf_counter() - lastRefreshTime < 1 and apb.under_rate_limit():
				src_files = []
				try:
					# generate some noise from 0.9 to 1.1 to apply to the files per job to attempt spreading out the job scheduling
					noise = random.uniform(0.9, 1.1)
					for _ in range(max(1,round(files_per_job * noise))):
						src_file = next(file_list_iterator)
						src_files.append(src_file)
					future = executor.submit(copy_files_bulk, src_files, dest_paths,src_path, full_hash, verbose,len(futures))
					futures[future] = src_files
					# counter += 1
				except StopIteration:
					if src_files:
						future = executor.submit(copy_files_bulk, src_files, dest_paths,src_path, full_hash, verbose,len(futures))
						futures[future] = src_files
					file_list_iterator = None

			done, _ = concurrent.futures.wait(futures.keys(), return_when=concurrent.futures.FIRST_COMPLETED,timeout=1)

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
				symLinks.update(rtnSymLinks)
			apb.scheduled_jobs = len(futures)
			if done:
				if verbose:
						print(f'\nAverage cptime is {current_iteration_total_run_time / len(done):0.2f} for {len(done)} jobs with {copied_file_count_this_run} files each')
				if file_list_iterator and copied_file_count_this_run == files_per_job and time.perf_counter() - lastRefreshTime > 1:
					files_per_job //= MAGIC_NUMBER
					files_per_job = round(files_per_job)
					if verbose:
						print(f'\nCompletion time is long, changing files per job to {files_per_job}')
				#elif file_list_iterator and copied_file_count_this_run == files_per_job and current_iteration_total_run_time / len(done) < 1:
				elif file_list_iterator and copied_file_count_this_run == files_per_job and time.perf_counter() - lastRefreshTime < 0.1:
					files_per_job *= MAGIC_NUMBER
					files_per_job = round(files_per_job)
					if verbose:
						print(f'\nCompletion time is short, changing files per job to {files_per_job}')
				if files_per_job < 1:
					files_per_job = 1
			else:
				if not apb.under_rate_limit():
					if verbose:
						print(f'\nWe had hit the rate limit, changing files per job to 1')
					files_per_job = 1
				time.sleep(apb.refresh_interval)
				apb.print_progress()
			lastRefreshTime = time.perf_counter()
	endTime = time.perf_counter()
	apb.stop()
	print(f"\nTime taken:             {endTime-start_time:0.4f} seconds")
	print(f"Average speed:          {format_bytes(apb.size_counter / (endTime-start_time))}B/s")
	print(f"                        {apb.item_counter / (endTime-start_time):.2f} file/s")
	print(f"Average bandwidth:      {format_bytes(apb.size_counter / (endTime-start_time) * 8,use_1024_bytes=False)}bps")
	return apb.item_counter, apb.size_counter , symLinks ,file_list

def get_copy_file_job_iter(jobs):
	for src_path, dest_paths, file_list in jobs:
		for f in file_list:
			yield (src_path, dest_paths, f)

def copy_file_list_parallel_batch(jobs, max_workers, full_hash=False, verbose=False, files_per_job=1,total_item_count=0,total_size_count=0):
	global FILES_RATE_LIMIT
	global BYTES_RATE_LIMIT
	if FILES_RATE_LIMIT or BYTES_RATE_LIMIT:
		max_scheduled_jobs = max_workers
	else:
		max_scheduled_jobs = max_workers * 1.1
	symLinks = {}
	# src_path, dest_paths, file_list
	job_list_iterator = get_copy_file_job_iter(jobs)
	start_time = time.perf_counter()
	lastRefreshTime = start_time
	futures = {}
	files_per_job = max(1,files_per_job)

	print(f"Processing {total_item_count} files with {max_workers} workers")
	apb = Adaptive_Progress_Bar(total_count=total_item_count,total_size=total_size_count,last_num_job_for_stats=max(1,max_workers//10),process_word='Copied',use_print_thread = True,suppress_all_output=verbose,bytes_rate_limit=BYTES_RATE_LIMIT,files_rate_limit=FILES_RATE_LIMIT)
	with ProcessPoolExecutor(max_workers=max_workers) as executor:
		while job_list_iterator or futures:
			# counter = 0
			while job_list_iterator and len(futures) < max_scheduled_jobs and time.perf_counter() - lastRefreshTime < 1 and apb.under_rate_limit():
				current_jobs = []
				try:
					# generate some noise from 0.9 to 1.1 to apply to the files per job to attempt spreading out the job scheduling
					noise = random.uniform(0.9, 1.1)
					for _ in range(max(1,round(files_per_job * noise))):
						job = next(job_list_iterator)
						current_jobs.append(job)
					future = executor.submit(copy_files_bulk_batch, current_jobs, full_hash, verbose,len(futures))
					futures[future] = current_jobs
					# counter += 1
				except StopIteration:
					if current_jobs:
						future = executor.submit(copy_files_bulk_batch, current_jobs, full_hash, verbose,len(futures))
						futures[future] = current_jobs
					job_list_iterator = None

			done, _ = concurrent.futures.wait(futures.keys(), return_when=concurrent.futures.FIRST_COMPLETED,timeout=1)

			#print('\n',len(done) ,'\t', len(futures))
			if job_list_iterator and len(done) > 1 and len(done) / len(futures) > 0.1:
				if verbose:
					print(f'\nTransfer is fast, doubling files per job to {files_per_job * 2}')
				files_per_job *= 2

			current_iteration_total_run_time = 0
			for future in done:
				current_jobs = futures.pop(future)
				copied_file_count_this_run = len(current_jobs)
				#try:
				cpSize, cpTime, rtnSymLinks = future.result()
				current_iteration_total_run_time += cpTime
				apb.update(num_files=copied_file_count_this_run,cpSize=cpSize,cpTime=cpTime,files_per_job=files_per_job)
				symLinks.update(rtnSymLinks)
			apb.scheduled_jobs = len(futures)
			if done:
				if verbose:
						print(f'\nAverage cptime is {current_iteration_total_run_time / len(done):0.2f} for {len(done)} jobs with {copied_file_count_this_run} files each')
				if job_list_iterator and copied_file_count_this_run == files_per_job and time.perf_counter() - lastRefreshTime > 1:
					files_per_job //= MAGIC_NUMBER
					files_per_job = round(files_per_job)
					if verbose:
						print(f'\nCompletion time is long, changing files per job to {files_per_job}')
				#elif file_list_iterator and copied_file_count_this_run == files_per_job and current_iteration_total_run_time / len(done) < 1:
				elif job_list_iterator and copied_file_count_this_run == files_per_job and time.perf_counter() - lastRefreshTime < 0.1:
					files_per_job *= MAGIC_NUMBER
					files_per_job = round(files_per_job)
					if verbose:
						print(f'\nCompletion time is short, changing files per job to {files_per_job}')
				if files_per_job < 1:
					files_per_job = 1
			else:
				if not apb.under_rate_limit():
					if verbose:
						print(f'\nWe had hit the rate limit, changing files per job to 1')
					files_per_job = 1
				time.sleep(apb.refresh_interval)
				apb.print_progress()
			lastRefreshTime = time.perf_counter()
	endTime = time.perf_counter()
	apb.stop()
	print(f"\nTime taken:             {endTime-start_time:0.4f} seconds")
	print(f"Average speed:          {format_bytes(apb.size_counter / (endTime-start_time))}B/s")
	print(f"                        {apb.item_counter / (endTime-start_time):.2f} file/s")
	print(f"Average bandwidth:      {format_bytes(apb.size_counter / (endTime-start_time) * 8,use_1024_bytes=False)}bps")
	return apb.item_counter, apb.size_counter , symLinks 

def copy_files_parallel(src_path, dest_paths, max_workers, full_hash=False, verbose=False,files_per_job=1,parallel_file_listing=False,exclude=None):
	# skip if src path or dest path is longer than 4096 characters
	if len(src_path) > 4096:
		print(f'Skipping: {src_path} is too long')
		return 0, 0 , set(), frozenset()
	newDests = []
	for dest in dest_paths:
		if len(dest) > 4096:
			print(f'\nSkipped {dest} because path is too long')
		else:
			newDests.append(dest)
	dest_paths = newDests
	if not dest_paths:
		print(f'\nSkipped {src_path} because all destination paths are too long')
		return 0, 0, set(), frozenset()
	if exclude and is_excluded(src_path,exclude):
		return 0, 0 , set(), frozenset()

	if not os.path.isdir(src_path):
		src_size, _ , symLinks = copy_file(src_path, dest_paths,full_hash=full_hash, verbose=verbose)
		return 1, src_size , symLinks , frozenset([src_path])
	start_time = time.perf_counter()
	if parallel_file_listing:
		file_list , links,init_size,folders  = get_file_list_parallel(src_path, max_workers,exclude=exclude)
	else:
		file_list,links,init_size,folders = get_file_list_serial(src_path,exclude=exclude)
		
	endTime = time.perf_counter()
	print(f"Time taken to get file list: {endTime-start_time:0.4f} seconds")
	total_files = len(file_list)
	print(f"Number of files: {total_files}")
	print(f"Number of links: {len(links)}")
	print(f"Number of folders: {len(folders)}")
	print(f"Estimated size: {format_bytes(init_size)}B")
	return copy_file_list_parallel(file_list=file_list,links=links,src_path=src_path, dest_paths=dest_paths, max_workers=max_workers, full_hash=full_hash, verbose=verbose,files_per_job=files_per_job,estimated_size = init_size)

def copy_files_parallel_batch(jobs, max_workers, full_hash=False, verbose=False,files_per_job=1,parallel_file_listing=False,exclude=None):
	newJobs = []
	total_symLinks = {}
	total_files = []
	copied_count = 0
	copied_size = 0
	total_item_count = 0
	total_size_count = 0
	total_link_count = 0
	total_folder_count = 0
	for src_path, dest_paths in jobs:
		# skip if src path or dest path is longer than 4096 characters
		if len(src_path) > 4096:
			print(f'Skipping: {src_path} is too long')
			continue
		newDests = []
		for dest in dest_paths:
			if len(dest) > 4096:
				print(f'\nSkipped {dest} because path is too long')
			else:
				newDests.append(dest)
		dest_paths = newDests
		if not dest_paths:
			print(f'\nSkipped {src_path} because all destination paths are too long')
			continue
		if exclude and is_excluded(src_path,exclude):
			continue
		if parallel_file_listing:
			file_list , links,init_size,folders  = get_file_list_parallel(src_path, max_workers,exclude=exclude)
		else:
			file_list,links,init_size,folders = get_file_list_serial(src_path,exclude=exclude)
		newJobs.append((src_path, dest_paths, file_list))
		total_item_count += len(file_list)
		total_size_count += init_size
		total_link_count += len(links)
		total_folder_count += len(folders)
		total_files.extend(file_list)
		for link in links:
			srcRelativePath = os.path.relpath(link, src_path)
			total_symLinks[link] = [os.path.join(dest_path,srcRelativePath) for dest_path in dest_paths]

	print(f"Number of files: {total_item_count}")
	print(f"Number of links: {total_link_count}")
	print(f"Number of folders: {total_folder_count}")
	print(f"Estimated size: {format_bytes(total_size_count)}B")
	item_counter, size_counter , symLinks =  copy_file_list_parallel_batch(newJobs, max_workers=max_workers, full_hash=full_hash, verbose=verbose,files_per_job=files_per_job,total_item_count=total_item_count,total_size_count=total_size_count)
	total_symLinks.update(symLinks)
	return total_item_count + item_counter, total_size_count + size_counter , total_symLinks ,frozenset(total_files)

def copy_files_serial(src_path, dest_paths, full_hash=False, verbose=False,exclude=None):
	# skip if src path or dest path is longer than 4096 characters
	global FILES_RATE_LIMIT
	global BYTES_RATE_LIMIT
	if len(src_path) > 4096:
		print(f'Skipping: {src_path} is too long')
		return 0, 0 , set(), frozenset()
	newDests = []
	for dest in dest_paths:
		if len(dest) > 4096:
			print(f'\nSkipped {dest} because path is too long')
		else:
			newDests.append(dest)
	dest_paths = newDests
	if not dest_paths:
		print(f'\nSkipped {src_path} because all destination paths are too long')
		return 0, 0 , set(), frozenset()
	if exclude and is_excluded(src_path,exclude):
		return 0, 0 , set(), frozenset()
	if not os.path.isdir(src_path):
		src_size, _ , symLinks = copy_file(src_path, dest_paths,full_hash=full_hash, verbose=verbose)
		return 1, src_size , symLinks , frozenset([src_path])
	print(f'Getting file list for {src_path}')
	file_list,links,init_size,_ = get_file_list_serial(src_path,exclude=exclude)
	links = set(links)
	total_files = len(file_list)
	print(f"Number of files: {total_files}")
	start_time = time.perf_counter()
	apb = Adaptive_Progress_Bar(total_count=total_files,total_size=init_size,last_num_job_for_stats=1,process_word='Copied',suppress_all_output=verbose,bytes_rate_limit=BYTES_RATE_LIMIT,files_rate_limit=FILES_RATE_LIMIT)
	for file in file_list:
		srcRelativePath = os.path.relpath(file, src_path)
		size, cpTime ,rtnSymLinks = copy_file(file, [os.path.join(dest_path, srcRelativePath) for dest_path in dest_paths],full_hash = full_hash, verbose=verbose)
		#update_progress_bar(copy_counter, copy_size_counter, total_files, start_time)
		apb.update(num_files=1,cpSize=size,cpTime=cpTime,files_per_job=1)
		links.update(rtnSymLinks)
		apb.rate_limit()
	symLinks = {}
	for link in links:
		srcRelativePath = os.path.relpath(link, src_path)
		symLinks[link] = [os.path.join(dest_path, srcRelativePath) for dest_path in dest_paths]
	endTime = time.perf_counter()
	apb.stop()
	print(f"\nTime taken:             {endTime-start_time:0.4f} seconds")
	print(f"Average speed:          {format_bytes(apb.size_counter / (endTime-start_time))}B/s")
	print(f"                        {apb.item_counter / (endTime-start_time):.2f} file/s")
	print(f"Average bandwidth:      {format_bytes(apb.size_counter / (endTime-start_time) * 8,use_1024_bytes=False)}bps")
	return apb.item_counter, apb.size_counter , symLinks , frozenset(file_list)

def copy_files_serial_batch(jobs, full_hash=False, verbose=False,exclude=None):
	# skip if src path or dest path is longer than 4096 characters
	global FILES_RATE_LIMIT
	global BYTES_RATE_LIMIT
	newJobs = []
	total_symLinks = {}
	total_files = []
	copied_count = 0
	copied_size = 0
	total_item_count = 0
	total_size_count = 0
	for src_path, dest_paths in jobs:
		if len(src_path) > 4096:
			print(f'Skipping: {src_path} is too long')
			continue
		newDests = []
		for dest in dest_paths:
			if len(dest) > 4096:
				print(f'\nSkipped {dest} because path is too long')
			else:
				newDests.append(dest)
		dest_paths = newDests
		if not dest_paths:
			print(f'\nSkipped {src_path} because all destination paths are too long')
			continue
		if exclude and is_excluded(src_path,exclude):
			continue
		if not os.path.isdir(src_path):
			src_size, _ , symLinks = copy_file(src_path, dest_paths,full_hash=full_hash, verbose=verbose)
			#return 1, src_size , symLinks , frozenset([src_path])
			copied_count += 1
			copied_size += src_size
			total_item_count += 1
			total_size_count += src_size
			total_symLinks.update(symLinks)
			total_files.append(src_path)
			continue
		print(f'Getting file list for {src_path}')
		file_list,links,init_size,_ = get_file_list_serial(src_path,exclude=exclude)
		newJobs.append((src_path, dest_paths, file_list, set(links), init_size))
		total_item_count += len(file_list)
		total_size_count += init_size
		total_files.extend(file_list)
	print(f"Number of files: {total_item_count}")
	print(f"Estimated size: {format_bytes(total_size_count)}B")
	if total_item_count == 0:
		return 0, 0, set(), frozenset()
	start_time = time.perf_counter()
	apb = Adaptive_Progress_Bar(total_count=total_item_count,total_size=total_size_count,last_num_job_for_stats=1,process_word='Copied',suppress_all_output=verbose,bytes_rate_limit=BYTES_RATE_LIMIT,files_rate_limit=FILES_RATE_LIMIT)
	apb.item_counter = copied_count
	apb.size_counter = copied_size
	for src_path, dest_paths, file_list, links, init_size in newJobs:
		for file in file_list:
			srcRelativePath = os.path.relpath(file, src_path)
			size, cpTime ,rtnSymLinks = copy_file(file, [os.path.join(dest_path, srcRelativePath) for dest_path in dest_paths],full_hash = full_hash, verbose=verbose)
			#update_progress_bar(copy_counter, copy_size_counter, total_files, start_time)
			apb.update(num_files=1,cpSize=size,cpTime=cpTime,files_per_job=1)
			links.update(rtnSymLinks)
			apb.rate_limit()
		
		for link in links:
			srcRelativePath = os.path.relpath(link, src_path)
			total_symLinks[link] = [os.path.join(dest_path, srcRelativePath) for dest_path in dest_paths]
	endTime = time.perf_counter()
	apb.stop()
	print(f"\nTime taken:             {endTime-start_time:0.4f} seconds")
	print(f"Average speed:          {format_bytes(apb.size_counter / (endTime-start_time))}B/s")
	print(f"                        {apb.item_counter / (endTime-start_time):.2f} file/s")
	print(f"Average bandwidth:      {format_bytes(apb.size_counter / (endTime-start_time) * 8,use_1024_bytes=False)}bps")
	return apb.item_counter, apb.size_counter , total_symLinks , frozenset(total_files)

class copy_scheduler:
	def __init__(self, max_workers = 4 * multiprocessing.cpu_count(), full_hash=False, verbose=False,files_per_job=1,parallel_file_listing=False,exclude=None):
		self.max_workers = max_workers
		self.full_hash = full_hash
		self.verbose = verbose	
		self.files_per_job = files_per_job
		self.parallel_file_listing = parallel_file_listing
		self.exclude = exclude
		self.dir_sync_job = []
		self.copy_job = []
		#copy_counter, copy_size_counter , rtnSymLinks , file_list 
		self.copy_counter = 0
		self.copy_size_counter = 0
		self.total_sym_links = {}
		self.total_file_list = []
	def add_dir_sync(self, src, dests):
		self.dir_sync_job.append((src, dests))
	def add_copy(self, src, dests):
		self.copy_job.append((src, dests))
	def process(self):
		if self.dir_sync_job:
			start_time = time.perf_counter()
			if self.max_workers == 1:
				self.total_sym_links.update(sync_directories_serial_batch(self.dir_sync_job,exclude=self.exclude))
			else:
				self.total_sym_links.update(sync_directories_parallel_batch(self.dir_sync_job, max_workers=self.max_workers,verbose=self.verbose,exclude=self.exclude))
			endTime = time.perf_counter()
			print(f"\nTime taken to sync directory: {endTime-start_time:0.4f} seconds")
			self.dir_sync_job = []
		if self.copy_job:
			global HASH_SIZE
			if HASH_SIZE == 0:
				print("Using file attributes only for skipping")
			elif xxhash_available:
				print("Using xxhash for skipping")
			else:
				print("Using blake2b for skipping")
			if self.max_workers == 1:
				copy_counter, copy_size_counter , rtnSymLinks , file_list = copy_files_serial_batch(self.copy_job, full_hash = self.full_hash,verbose=self.verbose,exclude=self.exclude)
				#total_file_list.update(trim_paths(file_list,src))
				#self.total_file_list.update(trim_paths(rtnSymLinks.keys(),src))
			else:
				copy_counter, copy_size_counter , rtnSymLinks , file_list = copy_files_parallel_batch(self.copy_job, max_workers=self.max_workers,full_hash = self.full_hash,verbose=self.verbose,files_per_job=self.files_per_job,parallel_file_listing=self.parallel_file_listing,exclude=self.exclude)
				#total_file_list.update(trim_paths(file_list,src))
				#self.total_file_list.update(trim_paths(rtnSymLinks.keys(),src))
			print(f'Total files copied:     {copy_counter}')
			print(f'Total size copied:      {format_bytes(copy_size_counter)}B')
			print(f'Total files discovered: {len(self.total_file_list)}')
			self.copy_counter += copy_counter
			self.copy_size_counter += copy_size_counter
			self.total_sym_links.update(rtnSymLinks)
			self.total_file_list.extend(file_list)
			self.copy_job = []
		return self.total_file_list, self.total_sym_links
	def clear(self,clear_pending_jobs=False):
		self.copy_counter = 0
		self.copy_size_counter = 0
		self.total_sym_links = {}
		self.total_file_list = []
		if clear_pending_jobs:
			self.dir_sync_job = []
			self.copy_job = []


#%% ---- Copy Directories ----
def sync_directory_metadata(src_path, dest_paths):
	# skip if src path or dest path is longer than 4096 characters
	if len(src_path) > 4096:
		print(f'Skipping: {src_path} is too long')
		return 0, 0 , set(), frozenset()
	newDests = []
	for dest in dest_paths:
		if len(dest) > 4096:
			print(f'\nSkipped {dest} because path is too long')
		else:
			newDests.append(dest)
	dest_paths = newDests
	if not dest_paths:
		print(f'\nSkipped {src_path} because all destination paths are too long')
		return 0, 0 , set(), frozenset()
	start_time = time.perf_counter()
	if os.path.islink(src_path):
		return 0,time.perf_counter()-start_time,{src_path: dest_paths}
	if not os.path.isdir(src_path):
		return copy_file(src_path, dest_paths)
	# Create the directory if it does not exist
	st = os.stat(src_path)
	for dest_path in dest_paths:
		try:
			if not (os.path.exists(dest_path) or os.path.ismount(dest_path)):
				os.makedirs(dest_path, exist_ok=True)
		except FileExistsError:
			print(f"Destination path {dest_path} maybe a mounted dir, known issue with os.path.exists\nContinuing without creating dest folder...")
		# Sync the metadata
		shutil.copystat(src_path, dest_path)
		if os.name == 'posix':
			os.chown(dest_path, st.st_uid, st.st_gid)
		os.utime(dest_path, (st.st_atime, st.st_mtime))
	return 1,time.perf_counter()-start_time,frozenset()

def sync_directory_metadata_bulk(src_paths, dest_paths,src_path):
	total_count = 0
	total_time = 0
	symLinks = {}
	for src in src_paths:
		source_relative_path = os.path.relpath(src, src_path)
		dests = [os.path.join(dest_path, source_relative_path) for dest_path in dest_paths]
		count , cpTime , rtnSymLinks = sync_directory_metadata(src, dests)
		total_count += count
		total_time += cpTime
		symLinks.update(rtnSymLinks)
	return total_count , total_time, symLinks

def sync_directory_metadata_bulk_batch(jobs):
	total_count = 0
	total_time = 0
	symLinks = {}
	for (src_path,folder,dest_paths) in jobs:
		source_relative_path = os.path.relpath(folder, src_path)
		dests = [os.path.join(dest_path, source_relative_path) for dest_path in dest_paths]
		count , cpTime , rtnSymLinks = sync_directory_metadata(folder, dests)
		total_count += count
		total_time += cpTime
		symLinks.update(rtnSymLinks)
	return total_count , total_time, symLinks

def sync_directories_parallel(src, dests, max_workers, verbose=False,folder_per_job=64,exclude=None):
	global FILES_RATE_LIMIT
	global BYTES_RATE_LIMIT
	# skip if src path or dest path is longer than 4096 characters
	if len(src) > 4096:
		print(f'Skipping: {src} is too long')
		return 0, 0 , set(), frozenset()
	newDests = []
	for d in dests:
		if len(d) > 4096:
			print(f'\nSkipped {d} because path is too long')
		else:
			newDests.append(d)
	dests = newDests
	if not dests:
		print(f'\nSkipped {src} because all destination paths are too long')
		return 0, 0, set(), frozenset()
	symLinks = {}
	if exclude and is_excluded(src,exclude):
		return 0, 0 , set(), frozenset()
	# 
	if not os.path.isdir(src):
		_, _ , symLinks = copy_file(src, dests)
		return symLinks
	sync_directory_metadata(src, dests)
	print(f'Getting file list for {src}')
	_,_,_,folders = get_file_list_serial(src,exclude=exclude)
	folder_list_iterator = iter(folders)
	futures = {}
	start_time = time.perf_counter()
	last_refresh_time = start_time
	max_workers = max(2,int(max_workers / 4))
	folder_per_job = max(1,folder_per_job)
	num_folders_copied_this_job = 0

	print(f"Syncing Dir from {src} to {dests} with {max_workers} workers")
	apb = Adaptive_Progress_Bar(total_count=len(folders),total_size=len(folders),use_print_thread = True,suppress_all_output=verbose,process_word='Synced',bytes_rate_limit=BYTES_RATE_LIMIT,files_rate_limit=FILES_RATE_LIMIT)
	with ProcessPoolExecutor(max_workers=max_workers) as executor:
		while folder_list_iterator or futures:
			# counter = 0
			while folder_list_iterator and len(futures) <  max_workers and last_refresh_time - time.perf_counter() < 5 and apb.under_rate_limit():
				src_folders = []
				try:
					for _ in range(folder_per_job):
						src_folder = next(folder_list_iterator)
						src_folders.append(src_folder)
					future = executor.submit(sync_directory_metadata_bulk, src_folders, dests,src)
					futures[future] = src_folders
					# counter += 1
				except StopIteration:
					if src_folders:
						future = executor.submit(sync_directory_metadata_bulk, src_folders, dests,src)
						futures[future] = src_folders
					folder_list_iterator = None

			done, _ = concurrent.futures.wait(futures.keys(), return_when=concurrent.futures.FIRST_COMPLETED,timeout=1)
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
				symLinks.update(rtnSymLinks)
			apb.scheduled_jobs = len(futures)
			if done:
				if verbose:
						print(f'\nAverage cptime is {time_spent_this_iter / len(done):0.2f} for {len(done)} jobs with {num_folders_copied_this_job} folders each')

				if folder_list_iterator and num_folders_copied_this_job == folder_per_job and (time_spent_this_iter / len(done) > 5 or time.perf_counter() - last_refresh_time > 5):
					folder_per_job //= MAGIC_NUMBER
					folder_per_job = round(folder_per_job)
					if verbose:
						print(f'\nCompletion time is long, changing folders per job to {folder_per_job}')
				elif folder_list_iterator and num_folders_copied_this_job == folder_per_job and time_spent_this_iter / len(done) < 1:
					folder_per_job *= MAGIC_NUMBER
					folder_per_job = round(folder_per_job)
					if verbose:
						print(f'\nCompletion time is short, changing folders per job to {folder_per_job}')
				if folder_per_job < 1:
					folder_per_job = 1
			else:
				if not apb.under_rate_limit():
					if verbose:
						print(f'\nWe had hit the rate limit, changing folder per job to 1')
					folder_per_job = 1
				time.sleep(apb.refresh_interval)
				apb.print_progress()
			apb.rate_limit()
			last_refresh_time = time.perf_counter()

	endTime = time.perf_counter()
	apb.stop()
	print(f"\nTime taken:             {endTime-start_time:0.4f} seconds")
	print(f"Average speed:          {format_bytes(apb.size_counter / (endTime-start_time))}B/s")
	print(f"                        {apb.item_counter / (endTime-start_time):.2f} folder/s")
	print(f"Average bandwidth:      {format_bytes(apb.size_counter / (endTime-start_time) * 8,use_1024_bytes=False)}bps")
	return symLinks

def sync_directories_parallel_batch(jobs, max_workers, verbose=False,folder_per_job=64,exclude=None):
	global FILES_RATE_LIMIT
	global BYTES_RATE_LIMIT
	symLinks = {}
	allFolderToSync = []
	for src, dests in jobs:
		# skip if src path or dest path is longer than 4096 characters
		if len(src) > 4096:
			print(f'Skipping: {src} is too long')
			continue
		newDests = []
		for d in dests:
			if len(d) > 4096:
				print(f'\nSkipped {d} because path is too long')
			else:
				newDests.append(d)
		dests = newDests
		if not dests:
			print(f'\nSkipped {src} because all destination paths are too long')
			continue
		if exclude and is_excluded(src,exclude):
			continue
		if not os.path.isdir(src):
			_, _ , sl = copy_file(src, dests)
			symLinks.update(sl)
			continue
		sync_directory_metadata(src, dests)
		print(f'Getting file list for {src}')
		_,_,_,folders = get_file_list_serial(src,exclude=exclude)
		allFolderToSync.extend([(src,folder,dests) for folder in folders])
	if not allFolderToSync:
		return symLinks
	print(f"Syncing Dir for {len(allFolderToSync)} folders with {max_workers} workers")
	apb = Adaptive_Progress_Bar(total_count=len(allFolderToSync),total_size=len(allFolderToSync),use_print_thread = True,suppress_all_output=verbose,process_word='Synced',bytes_rate_limit=BYTES_RATE_LIMIT,files_rate_limit=FILES_RATE_LIMIT)
	job_iterator = iter(allFolderToSync)
	futures = {}
	start_time = time.perf_counter()
	last_refresh_time = start_time
	max_workers = max(2,int(max_workers / 4))
	folder_per_job = max(1,folder_per_job)
	num_folders_copied_this_job = 0
	
	with ProcessPoolExecutor(max_workers=max_workers) as executor:
		while job_iterator or futures:
			while job_iterator and len(futures) <  max_workers and last_refresh_time - time.perf_counter() < 5 and apb.under_rate_limit():
				currentJobs = []
				try:
					for _ in range(folder_per_job):
						currentJobs.append(next(job_iterator))
					future = executor.submit(sync_directory_metadata_bulk_batch, currentJobs)
					futures[future] = currentJobs
				except StopIteration:
					if currentJobs:
						future = executor.submit(sync_directory_metadata_bulk_batch, currentJobs)
						futures[future] = currentJobs
					job_iterator = None

			done, _ = concurrent.futures.wait(futures.keys(), return_when=concurrent.futures.FIRST_COMPLETED,timeout=1)
			time_spent_this_iter = 0
			for future in done:
				currentJobs = futures.pop(future)
				num_folders_copied_this_job = len(currentJobs)
				#try:
				cpSize, cpTime, rtnSymLinks = future.result()
				time_spent_this_iter += cpTime
				#except Exception as exc:
					#print(f'\n{future} generated an exception: {exc}')
				#else:
				apb.update(num_files=num_folders_copied_this_job, cpSize=cpSize, cpTime=cpTime , files_per_job=folder_per_job)
				symLinks.update(rtnSymLinks)
			apb.scheduled_jobs = len(futures)
			if done:
				if verbose:
						print(f'\nAverage cptime is {time_spent_this_iter / len(done):0.2f} for {len(done)} jobs with {num_folders_copied_this_job} folders each')
				if job_iterator and num_folders_copied_this_job == folder_per_job and (time_spent_this_iter / len(done) > 5 or time.perf_counter() - last_refresh_time > 5):
					folder_per_job //= MAGIC_NUMBER
					folder_per_job = round(folder_per_job)
					if verbose:
						print(f'\nCompletion time is long, changing folders per job to {folder_per_job}')
				elif job_iterator and num_folders_copied_this_job == folder_per_job and time_spent_this_iter / len(done) < 1:
					folder_per_job *= MAGIC_NUMBER
					folder_per_job = round(folder_per_job)
					if verbose:
						print(f'\nCompletion time is short, changing folders per job to {folder_per_job}')
				if folder_per_job < 1:
					folder_per_job = 1
			else:
				if not apb.under_rate_limit():
					if verbose:
						print(f'\nWe had hit the rate limit, changing folder per job to 1')
					folder_per_job = 1
				time.sleep(apb.refresh_interval)
				apb.print_progress()
			apb.rate_limit()
			last_refresh_time = time.perf_counter()

	endTime = time.perf_counter()
	apb.stop()
	print(f"\nTime taken:             {endTime-start_time:0.4f} seconds")
	print(f"Average speed:          {format_bytes(apb.size_counter / (endTime-start_time))}B/s")
	print(f"                        {apb.item_counter / (endTime-start_time):.2f} folder/s")
	print(f"Average bandwidth:      {format_bytes(apb.size_counter / (endTime-start_time) * 8,use_1024_bytes=False)}bps")
	return symLinks

def sync_directories_serial(src, dests,exclude=None):
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
	global FILES_RATE_LIMIT
	global BYTES_RATE_LIMIT
	# skip if src path or dest path is longer than 4096 characters
	if len(src) > 4096:
		print(f'Skipping: {src} is too long')
		return 0, 0 , set(), frozenset()
	newDests = []
	for d in dests:
		if len(d) > 4096:
			print(f'\nSkipped {d} because path is too long')
		else:
			newDests.append(d)
	dests = newDests
	if not dests:
		print(f'\nSkipped {src} because all destination paths are too long')
		return 0, 0, set(), frozenset()
	if exclude and is_excluded(src,exclude):
		return 0, 0 , set(), frozenset()
	symLinks = {}
	if not os.path.isdir(src):
		_, _ , symLinks = copy_file(src, dests)
		return symLinks
	#sync_directory_metadata(src, dest)
	print(f'Getting file list for {src}')
	_,_,_,folders = get_file_list_serial(src,exclude=exclude)
	print(f"Syncing Dir from {src} to {dests} in single thread")
	apb = Adaptive_Progress_Bar(total_count=len(folders),total_size=len(folders),process_word='Synced',bytes_rate_limit=BYTES_RATE_LIMIT,files_rate_limit=FILES_RATE_LIMIT)
	for folder in folders:
		source_relative_path = os.path.relpath(folder, src)
		count , cpTime , _ = sync_directory_metadata(folder, [os.path.join(dest, source_relative_path) for dest in dests])
		apb.update(num_files=1, cpSize=count, cpTime=cpTime , files_per_job=1)
		apb.rate_limit()
	apb.stop()
	return symLinks

def sync_directories_serial_batch(jobs,exclude=None):
	global FILES_RATE_LIMIT
	global BYTES_RATE_LIMIT
	newJobs = []
	# skip if src path or dest path is longer than 4096 characters
	symLinks = {}
	totalFolderCount = 0
	srcFolders = []
	for src,dests in jobs:
		if len(src) > 4096:
			print(f'Skipping: {src} is too long')
			continue
		newDests = []
		for d in dests:
			if len(d) > 4096:
				print(f'\nSkipped {d} because path is too long')
			else:
				newDests.append(d)
		dests = newDests
		if not dests:
			print(f'\nSkipped {src} because all destination paths are too long')
			continue
		if exclude and is_excluded(src,exclude):
			continue
		if not os.path.isdir(src):
			_, _ , sls = copy_file(src, dests)
			symLinks.update(sls)
			continue
		newJobs.append((src,dests))
		print(f'Getting file list for {src}')
		_,_,_,folders = get_file_list_serial(src,exclude=exclude)
		srcFolders.append(folders)
		totalFolderCount += len(folders)
	if totalFolderCount == 0:
		return symLinks
	print(f"Syncing Dir from {len(newJobs)} in single thread")
	apb = Adaptive_Progress_Bar(total_count=totalFolderCount,total_size=totalFolderCount,process_word='Synced',bytes_rate_limit=BYTES_RATE_LIMIT,files_rate_limit=FILES_RATE_LIMIT)
	for (src, dests), folders in zip(newJobs, srcFolders):
		for folder in folders:
			source_relative_path = os.path.relpath(folder, src)
			count , cpTime , _ = sync_directory_metadata(folder, [os.path.join(dest, source_relative_path) for dest in dests])
			apb.update(num_files=1, cpSize=count, cpTime=cpTime , files_per_job=1)
			apb.rate_limit()
	apb.stop()
	return symLinks

#%% ---- Compare Files ----
def compare_file_list(file_list, file_list2,diff_file_list=None,tar_diff_file_list = False):
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

#%% ---- Remove Extra ----
def remove_extra_dirs(src_paths, dests,exclude=None):
	"""
	Removes extra directories in destination that are not present in the source paths.

	:param src_paths: list of source paths
	:param dest: destination path
	:return: None
	"""
	# Skip if the dest path is too long
	newDests = []
	for d in dests:
		if len(d) > 4096:
			print(f'\nSkipped {d} because path is too long')
		elif exclude and is_excluded(d,exclude):
			print(f'\nSkipped {d} because it is excluded')
		else:
			newDests.append(d)
	dests = newDests
	if not dests:
		print(f'\nSkipped {src_paths} because all destination paths are too long')
		return 0, 0, set(), frozenset()
	# remove excluded paths from src_paths
	for src_path in src_paths:
		if exclude and is_excluded(src_path,exclude):
			src_paths.remove(src_path)
	extraDirs = set()
	print(dests)
	for dest in dests:
		for dirpath, dirnames, _ in os.walk(dest, topdown=False):
			for dirname in dirnames:
				dirname += os.path.sep
				dest_dir_path = os.path.join(dirpath, dirname)
				# Check if the directory exists in the source paths
				if not any(os.path.exists(os.path.join(os.path.dirname(src_path), os.path.relpath(dest_dir_path, dest))) for src_path in src_paths):
					if exclude and is_excluded(dest_dir_path,exclude):
						print(f"Skipping excluded directory: {dest_dir_path}")
					elif os.path.ismount(dest_dir_path):
						print(f"Skipping mount point: {dest_dir_path}")
					else:
						print(f"Deleting extra directory: {dest_dir_path}")
						extraDirs.add(dest_dir_path)
	for dir in extraDirs:
		os.rmdir(dir)

def remove_extra_files(total_file_list, dests,max_workers,verbose,files_per_job,single_thread=False,exclude=None):
		print(f"Removing extra files from {dests} with {max_workers} workers")
		# we first get a file list of the dest dir
		inDestNotInSrc = set()
		for dest in dests:
			if single_thread:
				dest_file_list,links,_, _ = get_file_list_serial(dest,exclude=exclude)
			else:
				dest_file_list,links,_ ,_ = get_file_list_parallel(dest, max_workers,exclude=exclude)
			dest_file_list = trim_paths(dest_file_list,dest)
			dest_file_list.update(trim_paths(links,dest))
			# we then get the list of all extra files
			inDestNotInSrc.update([os.path.join(dest,file) for file in (dest_file_list - total_file_list)])
		print('-'*80)
		print(f"Files in dest but not in src:")
		for file in inDestNotInSrc:
			print(file)
		print('-'*80)
		if len(inDestNotInSrc) == 0:
			print(f"No extra files found in {dests}")
		else:
			print(f"Files in dest but not in src count: {len(inDestNotInSrc)}")
			print(f"Do you want to delete them? (y/n)")
			if not input().lower().startswith('y'):
				exit(0)
			start_time = time.perf_counter()
			if single_thread:
				for file in inDestNotInSrc:
					if os.path.isfile(file) or os.path.islink(file):
						os.remove(file)
					else:
						shutil.rmtree(file)
			else:
				delete_file_list_parallel(inDestNotInSrc, max_workers, verbose,files_per_job)
			endTime = time.perf_counter()
			print(f"Time taken to remove extra files: {endTime-start_time:0.4f} seconds")

#%% ---- Main Helper Functions ----
def mount_src_image(src_image,src_images: list,src_paths: list,mount_points: list,loop_devices: list):
	for src_image_pattern in src_image:
		if os.name != 'nt':
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
				run_command_in_multicmd_with_path_check(["mount","-o","ro",partition,target_mount_point],strict=False)
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
						run_command_in_multicmd_with_path_check(["mount","-o","ro",partition,target_mount_point],strict=False)
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

def verify_src_path(src_path,src_paths: list):
	if src_path:
		for src_path_pattern in src_path:
			#print(src_path_pattern)
			if os.name != 'nt':
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

def load_file_list(file_list):
	if not os.path.exists(file_list):
		print(f"File list {file_list} does not exist")
		return frozenset()
	with open(file_list, 'r') as f:
		fileList = frozenset([entry.strip() for entry in f.read().splitlines() if entry.strip()])
	return fileList

def store_file_list(file_list, src_paths: list, single_thread=False, max_workers=4 * multiprocessing.cpu_count(), verbose=False,
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
		file_list_file = load_file_list(file_list)
		if len(src_paths) == 1:
			remove_extra_files(file_list_file, src_paths[0], max_workers, verbose, files_per_job, single_thread, exclude=exclude)
			print('-' * 80)
		else:
			print("Currently only supports removing extra files for a single src_path when using file_list")
		exit(0)
	fileList = set()
	for src in src_paths:
		print(f"Getting file list from {src}")
		start_time = time.perf_counter()
		if not parallel_file_listing:
			files, links, _,folders = get_file_list_serial(src, exclude=exclude,append_hash=append_hash,full_hash=full_hash)
		else:
			files, links, _,folders  = get_file_list_parallel(src, max_workers, exclude=exclude,append_hash=append_hash,full_hash=full_hash)
		fileList.update(trim_paths(files, src))
		fileList.update(trim_paths(links, src))
		fileList.update([folder_path + os.path.sep for folder_path in trim_paths(folders, src)])
		endTime = time.perf_counter()
		print(f"Time taken to get file list: {endTime - start_time:0.4f} seconds")
	if compare_file_list:
		# This means we have a file_list and a src_path so we compare them
		print(f"Comparing file list from {src_paths} with {file_list}")
		if diff_file_list == 'auto':
			if not src_str:
				src_str = '-'.join([os.path.basename(os.path.realpath(src)) for src in src_paths])
			diff_file_list = f'DIFF_{src_str}_TO_{os.path.basename(os.path.realpath(file_list))}_{int(time.time())}_{"tar_" if tar_diff_file_list else ""}file_list.txt'
		fileList2 = load_file_list(file_list)
		compare_file_list(fileList, fileList2,diff_file_list,tar_diff_file_list = tar_diff_file_list)
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

def process_remove(src_paths: list,single_thread = False, max_workers = 4 * multiprocessing.cpu_count(),verbose = False,
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
		start_time = time.perf_counter()
		if single_thread:
			if os.path.isfile(path) or os.path.islink(path):
				if verbose:
					print(f"Removing file: {path}")
				try:
					os.remove(path)
				except Exception as e:
					print(f"Error removing file {path}: {e}")
			elif os.path.isdir(path):
				if verbose:
					print(f"Removing directory: {path}")
				try:
					shutil.rmtree(path)
				except Exception as e:
					print(f"Error removing directory {path}: {e}")
		else:
			delete_files_parallel(processedPaths if batch else path, max_workers, verbose=verbose,files_per_job=files_per_job,exclude=exclude,batch=batch)
		endTime = time.perf_counter()
		print(f"Time taken to remove files: {endTime-start_time:0.4f} seconds")
		print('-'*80)
		if batch and not single_thread:
			break

def get_dest_from_image(dest_image,mount_points: list,loop_devices: list):
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
		run_command_in_multicmd_with_path_check(["mount",target_partition,target_mount_point],strict=False)
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
					run_command_in_multicmd_with_path_check(["mount",target_partition,target_mount_point],strict=False)
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

def get_dest_from_path(dest_path,src_paths: list,src_path,can_be_none = False):
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
				dest = str(src_path[-1])
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
	if not dest:
		return None
	try:
		dest = str(dest)
	except Exception as e:
		print(f"Error converting dest_path {dest} to string, ignoring it")
		return None
	if len(src_paths) == 1 and os.path.isdir(src_paths[0]) and not src_paths[0].endswith(os.path.sep) and not dest.endswith(os.path.sep):
		src_paths[0] += os.path.sep
		dest += os.path.sep
	if (len(src_paths) > 1 or src_paths[0].endswith(os.path.sep)) and not dest.endswith(os.path.sep):
		dest += os.path.sep
	return dest

def get_dests(dest_paths,dest_image,mount_points: list,loop_devices: list,src_paths: list,src_path,can_be_none = False):
	dests = []
	target_mount_point = ''
	if dest_image:
		if os.geteuid() != 0:
			print(f"Warning: mounting dest image likely requires root privileges, please expect errors if continuing. continuing in 5 seconds...")
			time.sleep(5)
		imgDest , target_mount_point = get_dest_from_image(dest_image,mount_points,loop_devices)
		if imgDest:
			dests.append(imgDest)
		elif dest_paths:
			print(f"Destination image {dest_image} does not exist, using dest_paths {dest_paths}")
	if dest_paths:
		for dest in dest_paths:
			pathDest = get_dest_from_path(dest,src_paths,src_path,can_be_none=can_be_none)
			if pathDest:
				dests.append(pathDest)
	# get the str representation of the dests
	dest_str = '_'.join([os.path.basename(os.path.realpath(dest)) for dest in dests])
	if not dest_str:
		dest_str = 'undefined'
	return dests, dest_str, target_mount_point
	
def process_compare_file_list(src_paths: list, dests, max_workers = 4 * multiprocessing.cpu_count(),
							  parallel_file_listing = False,exclude=None,dest_image = None,diff_file_list = None,tar_diff_file_list = False,
							  append_hash = True,full_hash = False):
	# while os.path.basename(dest) == '':
	#     dest = os.path.dirname(dest)
	newDests = []
	for dest in dests:
		if dest and os.path.exists(dest) and os.path.isdir(dest):
			newDests.append(dest)
	dests = newDests
	if not dests:
		print(f"Destination image {dest_image} does not exist or dests {dests} is empty, exiting.")
		exit(1)
	print(f"Comparing file list from {src_paths} with {dests}")
	file_list = set()
	for src in src_paths:
		print('-'*80)
		print(f"Getting file list from {src}")
		start_time = time.perf_counter()
		if not parallel_file_listing:
			files,links,_,folders = get_file_list_serial(src,exclude=exclude,append_hash=append_hash,full_hash=full_hash)
		else:
			files,links,_,folders  = get_file_list_parallel(src, max_workers,exclude=exclude,append_hash=append_hash,full_hash=full_hash)
		if dest_image:
			# we use full path for src when comparing file list with dest image
			file_list.update(trim_paths([os.path.abspath(file) for file in files],'/'))
			file_list.update(trim_paths([os.path.abspath(link) for link in links],'/'))
			file_list.update([folder_path + os.path.sep for folder_path in trim_paths([os.path.abspath(folder) for folder in folders],'/')])
		else:
			file_list.update(trim_paths(files,src))
			file_list.update(trim_paths(links,src))
			file_list.update([folder_path + os.path.sep for folder_path in trim_paths(folders, src)])
		endTime = time.perf_counter()
		print(f"Time taken to get file list: {endTime-start_time:0.4f} seconds")
	start_time = time.perf_counter()
	file_list2 = set()
	print('-'*80)
	print(f"Getting file list from {dests}")
	for dest in dests:
		if not parallel_file_listing:
			files,links,_,folders = get_file_list_serial(dest,exclude=exclude,append_hash=append_hash,full_hash=full_hash)
		else:
			files,links,_,folders  = get_file_list_parallel(dest, max_workers,exclude=exclude,append_hash=append_hash,full_hash=full_hash)
		file_list2.update(trim_paths(files,dest))
		file_list2.update(trim_paths(links,dest))
		file_list2.update([folder_path + os.path.sep for folder_path in trim_paths(folders, dest)])
	endTime = time.perf_counter()
	print(f"Time taken to get file list: {endTime-start_time:0.4f} seconds")
	compare_file_list(file_list, file_list2, diff_file_list,tar_diff_file_list = tar_diff_file_list)

def create_image(dest_image,target_mount_point,loop_devices: list,src_paths: list,mount_points, max_workers = 4 * multiprocessing.cpu_count(),
				parallel_file_listing = False,exclude=None,dest_image_size=0):
	if target_mount_point and dest_image:
		# This means we were supplied a dest_image that does not exist, we need to create it and initialize it
		returnMountPoints = []
		currentMountPoint = target_mount_point + os.path.sep
		init_size = 0
		for src in src_paths:
			src = os.path.abspath(src + os.path.sep)
			if not parallel_file_listing:
				_,_,size,_ = get_file_list_serial(src,exclude=exclude)
				init_size += size
			else:
				_,_,size,_ = get_file_list_parallel(src, max_workers,exclude=exclude)
				init_size += size
		slag = 50*1024*1024
		image_file_size = int(1.05 *init_size + slag) # add 50 MB for the file system
		image_file_size = (int(image_file_size / 4096.0) + 1) * 4096 # round up to the nearest 4 KiB
		number_of_images = 1
		if dest_image_size <= 0:
			print(f"Estimated content size {format_bytes(init_size)}B Creating {dest_image} with size {format_bytes(image_file_size)}B")
		elif dest_image_size > image_file_size:
			print(f"Destination image size {format_bytes(dest_image_size)}B is larger than estimated content size {format_bytes(image_file_size)}B, using rounded {format_bytes(dest_image_size)}B")
			image_file_size = dest_image_size
		else:
			if slag >= dest_image_size:
				print(f"Estimated file system bloat size {format_bytes(slag)}B is larger than destination image size {format_bytes(dest_image_size)}B, exiting")
				exit(1)
			dest_image_usable_size = int((dest_image_size - slag) * 0.90)
			number_of_images = image_file_size // dest_image_usable_size + 1
			print(f"Destination image size {format_bytes(dest_image_size)}B is smaller than estimated content size {format_bytes(image_file_size)}B, creating {number_of_images} images of size {format_bytes(dest_image_size)}B")
			image_file_size = dest_image_size
		image_file_size = (int(image_file_size / 4096.0) + 1) * 4096 # round up to the nearest 4 KiB
		for i in range(number_of_images):
			if i > 0:
				if '.img' in dest_image:
					imageName = dest_image.replace('.img',f'_{i}.img')
				elif '.iso' in dest_image:
					imageName = dest_image.replace('.iso',f'_{i}.iso')
				else:
					imageName = dest_image + f'_{i}'
				currentMountPoint = tempfile.mkdtemp() + os.path.sep
				mount_points.append(currentMountPoint)
			else:
				imageName = dest_image
			try:
				# use truncate to allocate the space
				#run_command_in_multicmd_with_path_check(["fallocate","-l",str(image_file_size),imageName])
				run_command_in_multicmd_with_path_check(['truncate','-s',str(image_file_size),imageName],strict=True)
			except:
				# use python native method to allocate the space
				print("truncate not available, using python native method to allocate space")
				with open(imageName, 'wb') as f:
					f.seek(image_file_size-1)
					f.write(b'\0')
			# setup a loop device
			target_loop_device_dest = create_loop_device(imageName)
			loop_devices.append(target_loop_device_dest)
			# zero the superblocks
			print(f"Clearing {target_loop_device_dest} and create GPT partition table")
			run_command_in_multicmd_with_path_check(['dd','if=/dev/zero','of='+target_loop_device_dest,'bs=1M','count=16'])
			#run_command_in_multicmd_with_path_check(f"parted -s {target_loop_device_dest} mklabel gpt")
			#run_command_in_multicmd_with_path_check(['sgdisk','-Z',target_loop_device_dest])

			print(f"Loop device {target_loop_device_dest} created")
			target_partition = get_largest_partition(target_loop_device_dest) # should just return the loop device itself, but just in case.
			# format the partition
			# check if mkudffs is available and image file size is smaller than 8 TiB
			# if shutil.which('mkudffs') and image_file_size < 8 * 1024 * 1024 * 1024 * 1024:
			# 	print(f"Formatting {target_partition} as udf")
			# 	run_command_in_multicmd_with_path_check(f"mkudffs --utf8 --media-type=hd --blocksize=2048 --lvid=HPCP_disk_image --vid=HPCP_img --fsid=HPCP_img --vsid=HPCP_img {target_partition}")
			# format with xfs if it is available and size bigger then 300 MiB
			if shutil.which('mkfs.xfs') and image_file_size > 300 * 1024 * 1024:
				print(f"Formatting {target_partition} as xfs")
				run_command_in_multicmd_with_path_check(['mkfs.xfs','-f',target_partition])
			else:
				print(f"Formatting {target_partition} as ext4")
				run_command_in_multicmd_with_path_check(['mkfs.ext4','-F',target_partition])
			# mount the loop device to a temporary folder
			print(f"Mounting {target_partition} at {currentMountPoint}")
			run_command_in_multicmd_with_path_check(["mount",target_partition,currentMountPoint],strict=False)
			# verify mount
			if not os.path.ismount(currentMountPoint):
				print(f"Destination image {imageName} cannot be mounted, exiting.")
				exit(1)
			returnMountPoints.append(currentMountPoint)
		return returnMountPoints
	else:
		print(f"Destination path not specified, exiting.")
		exit(0)

def process_copy(src_paths: list, dests:list = [], single_thread = False, max_workers = 4 * multiprocessing.cpu_count(),verbose = False, 
				directory_only = False,no_directory_sync = False, full_hash = False, files_per_job = 1, parallel_file_listing = False,
				exclude=None,dest_image = None,batch = False):
	total_file_list = set()
	total_sym_links = {}
	taskCtr = 0
	argDest = dests.copy()
	if batch:
		print(f"Copying from {src_paths} to {dests}")
	max_workers = max_workers if not single_thread else 1
	cs = copy_scheduler(max_workers=max_workers,full_hash=full_hash,verbose=verbose,files_per_job=files_per_job,parallel_file_listing=parallel_file_listing,exclude=exclude)
	for src in src_paths:
		taskCtr += 1
		dests = argDest.copy()
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
			#destParentDirs = [ os.path.abspath(dest + srcParentDir + os.path.sep) for srcParentDir in srcParentDirs]
			sync_directory_metadata_bulk(srcParentDirs,dests,src_path='/')
			#dest = os.path.abspath(dest + src + os.path.sep)
			dests = [os.path.abspath(dest + src + os.path.sep) for dest in dests]
		else:
			sourceFolderName = os.path.basename(src)
			if sourceFolderName:
				src += os.path.sep
			src = os.path.abspath(src)
			newDests = []
			for dest in dests:
				if os.path.basename(dest) == '' and sourceFolderName:
					dest = os.path.join(dest,sourceFolderName)
					dest += os.path.sep
				newDests.append(os.path.abspath(dest))
			dests = newDests
		if not batch:
			print('-'*80)
			print(f"Task {taskCtr} of {len(src_paths)}, copying from {src} to {dests}")
		# verify dest is writable
		for dest in dests:
			if not os.access(os.path.dirname(os.path.abspath(dest)), os.W_OK):
				print(f"Destination {dest} is not writable, continue with caution.")
				#exit(1)
		if os.path.islink(src):
			total_sym_links[src] = dests
			print(f"{src} is a symlink, creating symlink in dests")
			continue
		# if os.path.isfile(src):
		# 	print("Copying single file")
		# 	copy_file(src, dests,full_hash=full_hash,verbose=verbose)
		# 	continue
		if os.path.isdir(src):
			src += os.path.sep
			if no_directory_sync:
				print("Skipping directory sync")
				sync_directory_metadata(src, dests)
			else:
				cs.add_dir_sync(src, dests)
		if not directory_only:
			cs.add_copy(src, dests)
		if not batch:
			fl, sl = cs.process()
			total_file_list.update(fl)
			total_sym_links.update(sl)
			cs.clear()
			print('-'*80)
	if batch:
		fl, sl = cs.process()
		total_file_list.update(fl)
		total_sym_links.update(sl)
		cs.clear()
		print(f"Processed {len(total_file_list)} files and {len(total_sym_links)} symlinks in total")
		print('-'*80)
	return total_file_list, total_sym_links

def validate_dd_source_path(src_path,loop_devices = None):
	if not loop_devices:
		loop_devices = []
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

def create_dd_dest_part_table(dd_src,dd_resize = [],src_path = None, dest_path = None):
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

def dd_partition(src_partition_path,dest_partition_path,partition,dd_src,dd_dest):
	# First verify the two partition size is the same
	src_part_info = get_partition_infos(dd_src)
	dest_part_info = get_partition_infos(dd_dest)
	if src_part_info[partition]['size'] > dest_part_info[partition]['size']:
		print(f"Source partition size {src_part_info[partition]['size']} is than the destination partition size {dest_part_info[partition]['size']}.")
		print(f"Cannot use DD, exiting.")
		exit(1)
	run_command_in_multicmd_with_path_check(['dd','if='+src_partition_path,'of='+dest_partition_path,'bs=1024M'],strict=True)

def clean_up(mount_points: list,loop_devices: list):
	# clean up loop devices and mount points if we are using a image
	for mount_point in mount_points:
		print(f"Unmounting {mount_point}")
		run_command_in_multicmd_with_path_check(["umount",mount_point])
		print(f"Removing mount point {mount_point}")
		os.rmdir(mount_point)
	for loop_device_dest in loop_devices:
		print(f"Removing loop device {loop_device_dest}")
		run_command_in_multicmd_with_path_check(['losetup','-d',loop_device_dest])

HASH_SIZE = 1<<24

def get_args(args = None):
	parser = argparse.ArgumentParser(description='Copy files from source to destination',
								  epilog=f'Found bins: {list(_binPaths.values())}\n Missing bins: {_binCalled - set(_binPaths.keys())}')
	parser.add_argument('-s', '--single_thread', action='store_true', help='Use serial processing')
	parser.add_argument('-j','-m','-t','--max_workers', type=int, default=4 * multiprocessing.cpu_count(), help='Max workers for parallel processing. Default is 4 * CPU count. Use negative numbers to indicate {n} * CPU count, 0 means 1/2 CPU count.')
	batch_group = parser.add_mutually_exclusive_group()
	batch_group.add_argument('-b','--batch',action='store_true', help='Batch mode, process all files in one go',default=True)
	batch_group.add_argument('-nb','--no_batch','--sequential',action='store_false', dest='batch', help='Do not use batch mode', default=False)
	parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
	parser.add_argument('-do', '--directory_only', action='store_true', help='Only copy directory structure')
	parser.add_argument('-nds', '--no_directory_sync', action='store_true', help='Do not sync directory metadata, useful for verfication')
	parser.add_argument('-fh', '--full_hash', action='store_true', help='Checks the full hash of files')
	parser.add_argument('-hs', '--hash_size', type=int, default=1<<16, help='Hash size in bytes, default is 65536. This means hpcp will only check the last 64 KiB of the file.')
	parser.add_argument('-fpj', '--files_per_job', type=int, default=1, help='Base number of files per job, will be adjusted dynamically. Default is 1')
	parser.add_argument('-sfl','-lfl', '--source_file_list', type=str, help='Load source file list from file. Will treat it raw meaning do not expand files / folders. files are seperated using newline.  If --compare_file_list is specified, it will be used as source for compare')
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
	parser.add_argument('-V', '--version', action='version', version=f"%(prog)s {version} @ {COMMIT_DATE} with {('XXHash' if xxhash_available else 'Blake2b')}, multiCMD V{multiCMD.version}, and [ {', '.join(_binPaths.keys())} ]; High Performance CoPy (HPC coPy) by pan@zopyr.us")
	parser.add_argument('-pfl', '--parallel_file_listing', action='store_true', help='Use parallel processing for file listing')
	parser.add_argument('src_path', nargs='*', type=str, help='Source Path')
	parser.add_argument('-si','--src_image',action='append', type=str, help='Source Image, mount the image and copy the files from it.')
	parser.add_argument('-siff','--load_diff_image',action='append', type=str, help='Not implemented. Load diff images and apply the changes to the destination.')
	parser.add_argument('-d','-C','--dest_path',action='append', type=str, help='Destination Path')
	parser.add_argument('-rds','--random_dest_selection', action='store_true', help='Randomly select destination path from the list of destination paths instead of filling round robin. Can speed up transfer if dests are on different devices. Warning: can cause unable to fit in big files as dests are filled up by smaller files.')
	parser.add_argument('-di','--dest_image', type=str, help='Base name for destination Image, create a image file and copy the files into it.')
	parser.add_argument('-dis','--dest_image_size', type=str, help=f'Destination Image Size, specify the size of the destination image to split into. Default is 0 (No split). Example: {{10TiB}} or {{1G}}', default='0')
	parser.add_argument('-diff', '--get_diff_image', action='store_true', help='Not implemented. Compare the source and destination file list, create a diff image of that will update the destination to source.')
	parser.add_argument('-dd', '--disk_dump', action='store_true', help='Disk to Disk mirror, use this if you are backuping / deploying an OS from / to a disk. \
					 Require 1 source, can be 1 src_path or 1 -si src_image, require 1 -di dest_image. Note: will only actually use dd if unable to mount / create a partition.')
	parser.add_argument('-ddr', '--dd_resize', action='append', type=str, help=f'Resize the destination image to the specified size with -dd. Applies to biggest partiton first. Specify multiple -ddr to resize subsequent sized partitions. Example: {{100GiB}} or {{200G}}')
	parser.add_argument('-L','-rl','--rate_limit', type=str, default=None, help='Approximate a rate limit the copy speed in bytes/second. Example: 10M for 10 MB/s, 1Gi for 1 GiB/s. Note: do not work in single thread mode. Default is 0: no rate limit.')
	parser.add_argument('-F','-frl','--file_rate_limit', type=str, default=None, help='Approximate a rate limit the copy speed in files/second. Example: 10K for 10240 files/s, 1Mi for 1024*1024*1024 files/s. Note: do not work in serial mode. Default is 0: no rate limit.')

	try:
		args = parser.parse_intermixed_args(args)
	except Exception:
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

#%% ---- Main Function ----
def hpcp(src_path, dest_paths = [], single_thread = False, max_workers = 4 * multiprocessing.cpu_count(),
			verbose = False, directory_only = False,no_directory_sync = False, full_hash = False, files_per_job = 1, target_file_list = "",
			compare_file_list = False, diff_file_list = None, tar_diff_file_list = False, remove = False,remove_force = False, remove_extra = False, parallel_file_listing = False,
			exclude=None,exclude_file = None,dest_image = None,dest_image_size = '0', no_link_tracking = False,src_image = None,dd = False,dd_resize = 0,
			batch = False, append_hash_to_file_list = True, hash_size = ..., source_file_list = None, random_destination_selection = False, bytes_rate_limit = None, files_rate_limit = None):
	global HASH_SIZE
	global RANDOM_DESTINATION_SELECTION
	global BYTES_RATE_LIMIT
	global FILES_RATE_LIMIT
	if random_destination_selection:
		RANDOM_DESTINATION_SELECTION = True
		print("Random destination selection enabled.")
	else:
		RANDOM_DESTINATION_SELECTION = False
	if bytes_rate_limit:
		BYTES_RATE_LIMIT = format_bytes(bytes_rate_limit,to_int=True)
	if files_rate_limit:
		FILES_RATE_LIMIT = format_bytes(files_rate_limit,to_int=True)
	if hash_size != ...:
		try:
			HASH_SIZE = int(hash_size)
		except:
			print(f"Invalid hash size {hash_size}, using default hash size {HASH_SIZE}")
	if HASH_SIZE < 0:
		HASH_SIZE = 0
	if HASH_SIZE == 0:
		print("Warning: Hash size set to 0, will not check file content for skipping.")
	try:
		dest_image_size = format_bytes(dest_image_size,to_int=True)
	except:
		print(f"Invalid destination image size {dest_image_size}, using default size 0")
		dest_image_size = 0
	print('-'*80)
	src_paths = []
	src_images = []
	mount_points = []
	loop_devices = []
	src_str = ''
	programStartTime = time.perf_counter()
	exclude = format_exclude(exclude,exclude_file)
	if max_workers == 0:
		max_workers = round(0.5 * multiprocessing.cpu_count())
	elif max_workers < 0:
		max_workers = round(- max_workers * multiprocessing.cpu_count())

	if dd:
		if os.name == 'nt':
			print("dd mode is not supported on Windows, exiting")
			return(0)
		# check if running as root
		if os.geteuid() != 0:
			print("WARNING: dd mode likely requires root privileges, please expect errors. Continuing in 5 seconds...")
			time.sleep(5)
		print("dd mode enabled, performing Disk Dump Copy. Setting up the target ...")
		if dest_paths:
			if len(dest_paths) > 1:
				print(f"Destination path is not 1, taking the first destination path as image file.")
			dest_path = dest_paths[0]
		else:
			dest_path = dest_image
		src_path = src_path if src_path else src_image

		if dest_image_size:
			print(f"Currently not supporting dest_image_size in dd mode. Ignoring dest_image_size {dest_image_size}.")
			
		if not dest_path and src_path:
			print(f"Destination path not specified, using {src_path[-1]} as destination")
			dest_path = src_path.pop()

		# check write permission on dest_path
		if not os.access(os.path.dirname(os.path.abspath(dest_path)), os.W_OK):
			print(f"Destination path {dest_path} is not writable, continuing with high probability of failure.")
			#exit(1)
		dd_src = validate_dd_source_path(src_path,loop_devices = loop_devices)
		partition_infos = create_dd_dest_part_table(dd_src,dd_resize=dd_resize,src_path=src_path, dest_path=dest_path)
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
		
		_ = partition_infos.pop(dd_src)
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
			if not any(os.scandir(src_mount_point)) and not os.path.ismount(src_mount_point) and is_device(src_partition_path):
				# if the source is a device and currently mounted, try to find the current mount point, then try to bind mount it
				mtab = get_mount_table()
				if src_partition_path in mtab:
					src_partition_mount_path = mtab[src_partition_path][0]
					print(f"Error mounting {src_partition_path}, trying to use bind mount {src_partition_mount_path}")
					# try to bind mount
					run_command_in_multicmd_with_path_check(['mount','--bind',src_partition_mount_path,src_mount_point],strict=False)
			if not any(os.scandir(src_mount_point)) and not os.path.ismount(src_mount_point):
				print(f"Error mounting {src_partition_path}, usig dd for copying.")
				dd_partition(src_partition_path,dest_partition_path,partition,dd_src,dd_dest)
				continue
			print(f"Mounting {dest_partition_path} at {dest_mount_point}")
			run_command_in_multicmd_with_path_check(['mount',dest_partition_path,dest_mount_point],strict=False)
			# check if the mount is successful
			if not any(os.scandir(dest_mount_point)) and not os.path.ismount(dest_mount_point) and is_device(dest_partition_path):
				mtab = get_mount_table()
				if dest_partition_path in mtab:
					dest_partition_mount_path = mtab[dest_partition_path][0]
					print(f"Error mounting {dest_partition_path}, trying to use bind mount {dest_partition_mount_path}")
					# try to bind mount
					run_command_in_multicmd_with_path_check(['mount','--bind',dest_partition_mount_path,dest_mount_point],strict=False)
			if not any(os.scandir(dest_mount_point)) and not os.path.ismount(dest_mount_point):
				print(f"Error mounting {dest_partition_path}, usig dd for copying.")
				dd_partition(src_partition_path,dest_partition_path,partition,dd_src,dd_dest)
				continue
			# copy the partition files
			print(f"Copying partition {partition} files from {src_mount_point} to {dest_mount_point}")
			hpcp([src_mount_point], dest_paths = [dest_mount_point], single_thread=single_thread, max_workers=max_workers,
																verbose=verbose, directory_only=directory_only,no_directory_sync=no_directory_sync,
																full_hash=full_hash, files_per_job=files_per_job, parallel_file_listing=parallel_file_listing,
																exclude=exclude,no_link_tracking = True)
		clean_up(mount_points,loop_devices)
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
		if os.geteuid() != 0:
			print("Warning: Source image mount may require root privileges, please expect errors. continuing in 5 seconds...")
			time.sleep(5)
		src_str = mount_src_image(src_image,src_images,src_paths,mount_points,loop_devices)

	if source_file_list:
		src_paths.extend(load_file_list(source_file_list))

	verify_src_path(src_path,src_paths)
	if not src_str:
		src_str = "-".join([os.path.basename(src) for src in src_paths])

	if target_file_list:
		store_file_list(target_file_list,src_paths,single_thread=single_thread, max_workers=max_workers,verbose=verbose, 
						files_per_job=files_per_job,compare_file_list=compare_file_list, remove_extra=remove_extra,
						parallel_file_listing=parallel_file_listing,exclude=exclude,diff_file_list=diff_file_list,tar_diff_file_list=tar_diff_file_list,src_str = src_str,
						append_hash=append_hash_to_file_list,full_hash=full_hash)
		clean_up(mount_points,loop_devices)
		return 0
	
	# if dest_image:
	# 	dest , target_mount_point = get_dest_from_image(dest_image,mount_points,loop_devices)
	# 	dest_str = dest_image
	# else:
	# 	dest = get_dest_from_path(dest_path,src_paths,src_path,can_be_none=remove)
	# 	dest_str = dest

	dests, dest_folder_name, target_mount_point = get_dests(dest_paths,dest_image,mount_points,loop_devices,src_paths,src_path,can_be_none = remove)

	if compare_file_list or diff_file_list:
		if diff_file_list == 'auto':
			if not src_str:
				src_str = '-'.join([os.path.basename(os.path.realpath(src)) for src in src_paths])
			diff_file_list = f'DIFF_{src_str}_TO_{dest_folder_name}_{int(time.time())}_{"tar_" if tar_diff_file_list else ""}file_list.txt'
		process_compare_file_list(src_paths, dests, max_workers=max_workers,parallel_file_listing=parallel_file_listing,
						 exclude=exclude,dest_image=dest_image,diff_file_list=diff_file_list,tar_diff_file_list=tar_diff_file_list,append_hash=append_hash_to_file_list,full_hash=full_hash)
		clean_up(mount_points,loop_devices)
		return 0
	
	for dest in dests:
		try:
			if dest and dest != ... and dest.endswith(os.path.sep) and (not (os.path.exists(dest) or os.path.ismount(dest))):
				os.makedirs(dest, exist_ok=True)
		except FileExistsError :
			print(f"Destination path {dest} maybe a mounted dir, known issue with os.path.exists\nContinuing without creating dest folder...")

	if not dests:
		if remove:
			dests = ...
		else:
			dests = create_image(dest_image,target_mount_point,loop_devices,src_paths,mount_points,max_workers=max_workers,parallel_file_listing=parallel_file_listing,exclude=exclude,dest_image_size = dest_image_size)

	if dests != ...:
		total_file_list, total_sym_links = process_copy(src_paths, dests, single_thread=single_thread, max_workers=max_workers,
																		verbose=verbose, directory_only=directory_only,no_directory_sync=no_directory_sync,
																		full_hash=full_hash, files_per_job=files_per_job, parallel_file_listing=parallel_file_listing,
																		exclude=exclude,dest_image=dest_image,batch = batch)
		if verbose:
			# sort file list and sym links
			for file in natural_sort(total_file_list):
				print(f"Copied File: {file}")
			for link in total_sym_links:
				print(f"Link: {link} -> {total_sym_links[link]}")
		create_sym_links(total_sym_links,exclude=exclude,no_link_tracking=no_link_tracking)
		if remove_extra:
			print('-'*80)
			remove_extra_files(total_file_list, dests,max_workers,verbose,files_per_job,single_thread,exclude=exclude)
			print('-'*80)
			print("Removing extra empty directories...")
			remove_extra_dirs(src_paths, dests,exclude=exclude)
		print('-'*80)
		if len(src_paths) > 1:
			print("Overall Summary:")
			print(f"Number of files / links: {len(total_file_list)}")
			print(f"Number of links: {len(total_sym_links)}")
			print(f"Total time taken: {time.perf_counter()-programStartTime:0.4f} seconds")
	if remove:
		process_remove(src_paths,single_thread=single_thread, max_workers=max_workers,verbose=verbose,
					  files_per_job=files_per_job, remove_force=remove_force,exclude=exclude,batch=batch)
	clean_up(mount_points,loop_devices)
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

#%% ---- CLI ----
def main():
	global RANDOM_DESTINATION_SELECTION
	args = get_args()
	# we run gui if the current platform is windows and src_path is not specified
	if os.name == 'nt' and len(args.src_path) == 0:
		hpcp_gui()
	else:
		rtnCode = hpcp(args.src_path, dest_paths = args.dest_path, single_thread = args.single_thread, max_workers = args.max_workers, verbose = args.verbose,
			 directory_only =  args.directory_only, no_directory_sync = args.no_directory_sync,full_hash = args.full_hash, files_per_job = args.files_per_job,
			 target_file_list = args.target_file_list, compare_file_list = args.compare_file_list , diff_file_list = args.diff_file_list, tar_diff_file_list = args.tar_diff_file_list,remove = args.remove, remove_force =args.remove_force,
			 remove_extra = args.remove_extra, parallel_file_listing = args.parallel_file_listing,exclude = args.exclude,exclude_file = args.exclude_file,
			 dest_image = args.dest_image,dest_image_size=args.dest_image_size,no_link_tracking = args.no_link_tracking,src_image = args.src_image,dd=args.disk_dump,
			 dd_resize=args.dd_resize,batch=args.batch,append_hash_to_file_list=not args.no_hash_file_list, hash_size=args.hash_size,source_file_list=args.source_file_list,random_destination_selection = args.random_dest_selection,bytes_rate_limit = args.rate_limit,files_rate_limit = args.file_rate_limit)
		if rtnCode:
			exit(rtnCode)
		

if __name__ == '__main__':
	main()