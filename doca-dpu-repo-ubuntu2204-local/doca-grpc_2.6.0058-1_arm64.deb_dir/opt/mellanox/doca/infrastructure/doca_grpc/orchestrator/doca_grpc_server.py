#!/usr/bin/python3

#
# Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES, ALL RIGHTS RESERVED.
#
# This software product is a proprietary product of NVIDIA CORPORATION &
# AFFILIATES (the "Company") and all right, title, and interest in and to the
# software product, including all associated intellectual property rights, are
# and shall remain exclusively with the Company.
#
# This software product is governed by the End User License Agreement
# provided with the software product.
#

import sys
import argparse
import logging
import shlex
import concurrent.futures as futures
import threading
import subprocess
import grpc
import os
import signal
import time
import configparser

import common_pb2 as gen_common
import common_pb2_grpc as gen_orchestration
import doca_grpc_orchestrator_pb2 as gen_pbuf
import doca_grpc_orchestrator_pb2_grpc as gen_grpc

CONFIG_FILE_PATH             = '/etc/doca_grpc/doca_grpc.conf'
CONFIG_FILE_SECTION_PROGRAMS = 'DOCA gRPC Programs'
CONFIG_FILE_KEY_ADDRESS      = 'server_address'
CONFIG_FILE_KEY_PROGRAMS     = 'programs'

NAME_TO_PORT = 	{
			# Infrastructure
			'doca_flow_grpc':		gen_common.eNetworkPort.k_DocaFlow,
		}

TERMINATION_CLEANUP_PERIOD_SECONDS  = 5
TERMINATION_GRACE_PERIOD_SECONDS    = 5
LIFETIME_CHECK_NUM_RETRIES          = 3
LIFETIME_CHECK_SLEEP_PERIOD_SECONDS = 10
SERVICE_NAME = 'DOCA gRPC Orchestrator Service'

DEFAULT_gRPC_PORT = gen_common.eNetworkPort.k_DocaGrpcOrchestrator

logger = logging.getLogger(SERVICE_NAME)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(ch)
logger.setLevel(logging.INFO)

all_known_programs = {}
known_programs_db = {}
server_addresses = []

def load_config():
	"""Parse the service's configuration.

	Raises:
		FileNotFoundError: If the config file is missing
		ValueError: If the config file is malformed
	"""
	global all_known_programs, known_programs_db, server_addresses

	all_known_programs = {}
	known_programs_db = {}
	server_addresses = []
	seen_addresses = set()

	config = configparser.ConfigParser()
	if not os.path.exists(CONFIG_FILE_PATH):
		raise FileNotFoundError(f'Missing config file at: {CONFIG_FILE_PATH}')

	try:
		config.read(CONFIG_FILE_PATH)
	except configparser.Error as e:
		raise ValueError(f'Errors while parsing configuration file: {CONFIG_FILE_PATH}')

	if not config.has_section(CONFIG_FILE_SECTION_PROGRAMS):
		raise ValueError(f'Config file missing section: {CONFIG_FILE_SECTION_PROGRAMS}')

	all_known_programs = config[CONFIG_FILE_SECTION_PROGRAMS]
	if len(all_known_programs) == 0:
		raise ValueError(f'Empty list of known programs')

	if len(config.sections()) == 1:
		raise ValueError('Missing server details, nothing to be done')

	for section_name in config.sections():
		section = config[section_name]
		if section_name == CONFIG_FILE_SECTION_PROGRAMS:
			continue
		for needed_key in [CONFIG_FILE_KEY_ADDRESS, CONFIG_FILE_KEY_PROGRAMS]:
			if needed_key not in section:
				raise ValueError(f'Section "{section_name}" missing key: {needed_key}')

		address_parts = section[CONFIG_FILE_KEY_ADDRESS].split(':')
		if len(address_parts) > 2:
			raise ValueError(f'Malformed value for {CONFIG_FILE_KEY_ADDRESS}: "{section[CONFIG_FILE_KEY_ADDRESS]}"')
		if len(address_parts) == 1:
			address_parts.append(str(DEFAULT_gRPC_PORT))
		if address_parts[0] in seen_addresses:
			raise ValueError(f'Multiple references to same server address: {address_parts[0]}')

		seen_addresses.add(address_parts[0])
		server_addresses.append(address_parts)
		server_uid = ':'.join(address_parts)

		server_program_parts = [x.strip() for x in section[CONFIG_FILE_KEY_PROGRAMS].split(',')]
		server_program_parts = [x for x in server_program_parts if len(x) > 0]
		if len(server_program_parts) == 0:
			raise ValueError(f'Missing list of programs for server {server_uid}')
		unknown_programs = set(server_program_parts) - set(all_known_programs.keys())
		if len(unknown_programs) != 0:
			raise ValueError(f'Reference to unknown programs for server {server_uid}: {", ".join(unknown_programs)}')
		known_programs_db[server_uid] = server_program_parts

class DocaOrchestrator(gen_grpc.DocaOrchestrator):
	def init(self, ip_addr, port):
		self.ip_addr = ip_addr
		self.uid = f'{ip_addr}:{port}'
		self.clients_db = {}

	def GetProgramList(self, request, context):
		"""Fetch the program list from the configuration file.

		Args:
			request (grpc): Incoming grpc request
			context (grpc): gRPC context

		Returns:
			DOCA ProgramList message
		"""
		logger.debug(f'{self.uid} - GetProgramList() - {", ".join(known_programs_db[self.uid])}')
		return gen_pbuf.ProgramList(program_names=known_programs_db[self.uid])

	def Create(self, request, context):
		"""Create a gRPC program process.

		Args:
			request (grpc): Incoming grpc request
			context (grpc): gRPC context

		Returns:
			DOCA RichStatus message
		"""
		logger.debug(f"{self.uid} - Create()")

		rich_status = gen_pbuf.RichStatus(err_status=gen_pbuf.Status(is_error=True))

		if request.program_name not in known_programs_db[self.uid]:
			rich_status.err_status.error_msg = f'Unknown program name: {request.program_name}'
			return rich_status
		process_path = all_known_programs[request.program_name]

		if request.port == 0:
			if request.program_name not in NAME_TO_PORT:
				rich_status.err_status.error_msg = f'No known default network port for: {request.program_name}'
				return rich_status
			network_port = NAME_TO_PORT[request.program_name]
		else:
			network_port = request.port

		process_uid = f'{self.ip_addr}:{network_port}'
		if process_uid in self.clients_db:
			rich_status.err_status.error_msg = f'Process UID {process_uid} already exists'
			return rich_status

		request_args = request.cmdline.split(' ')
		cmd_args  = [process_path]
		cmd_args += request_args
		# Checking for json mode
		if len(request_args) != 2 or request_args[0] not in ["-j", "--json"]:
			cmd_args += ['--grpc-address']
			cmd_args += [process_uid]
		proc = subprocess.Popen(cmd_args)

		logger.info(f'Connecting to DOCA Orchestration of {request.program_name} at: {process_uid}')
		channel = grpc.insecure_channel(f'{process_uid}')
		stub = gen_orchestration.DocaOrchestrationStub(channel)

		# Attempt to connect to it before we send back the status to the host
		spawn_success = False
		for i in range(LIFETIME_CHECK_NUM_RETRIES):
			try:
				stub.HealthCheck(gen_common.HealthCheckReq(), timeout=LIFETIME_CHECK_SLEEP_PERIOD_SECONDS,
					wait_for_ready=True)
				self.clients_db[process_uid] = (stub, proc)
				rich_status.err_status.is_error = False
				spawn_success = True
				break
			except grpc._channel._InactiveRpcError as e:
				proc.poll()
				if proc.returncode is not None:
					break
				logger.warning(f'Waiting for {process_uid} to spawn - Attempt #{i+1}')

		if not spawn_success:
			rich_status.err_status.is_error = True
			rich_status.err_status.error_msg = f'Failed to spawn {process_uid}!'
			logger.warning(rich_status.err_status.error_msg)

		rich_status.uid.uid = process_uid
		return rich_status

	def Destroy(self, request, context):
		"""Destroy a running gRPC program.

		Args:
			request (grpc): Incoming grpc request
			context (grpc): gRPC context

		Returns:
			DOCA Status message
		"""
		logger.debug(f'{self.uid} - Destroy()')

		if request.uid not in self.clients_db:
			return gen_pbuf.Status(is_error=True, error_msg=f'Unknown Process UID: {request.uid}')

		client, proc = self.clients_db[request.uid]
		# Start by removing it from the mapping.
		# gRPC exceptions aren't always caught below, even when the program isn't running.
		self.clients_db.pop(request.uid)
		try:
			client.Destroy(gen_common.DestroyReq())
			error_msg = None
		except grpc._channel._InactiveRpcError as e:
			error_msg = f'Failed to destroy {request.uid}. Is it really running?'
			logger.error(error_msg)

		# Wait a bit and let the program shut itself down.
		# When the timer expires, terminate it forcefully
		for i in range(2):
			try:
				proc.wait(timeout=TERMINATION_GRACE_PERIOD_SECONDS)
				break
			except subprocess.TimeoutExpired as e:
				proc.terminate()

		# Sample the process's return code
		proc.poll()
		if proc.returncode is None:
			error_msg = f'Failed to destroy {request.uid}!'
			logger.warning(error_msg)
		else:
			# Even after the process terminates, the OS needs time to clear all the resources
			time.sleep(TERMINATION_CLEANUP_PERIOD_SECONDS)

		return gen_pbuf.Status(is_error=error_msg is not None, error_msg=error_msg)

def start_server_instance(ip_addr, port):
	"""Start a gRPC server instance.

	Args:
		ip_addr (str): IP address for the bound server
		port (str): TCP port to be used by the server
	"""
	logger.info(f'Starting a gRPC server that listens for requests from the host: {ip_addr}:{port}')
	grpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
	service_instance = DocaOrchestrator()
	service_instance.init(ip_addr, port)
	gen_grpc.add_DocaOrchestratorServicer_to_server(service_instance, grpc_server)
	try:
		grpc_server.add_insecure_port(f'{ip_addr}:{port}')
	except RuntimeError as e:
		logger.error(f'Failed binding server address {ip_addr}:{port}')
		return
	grpc_server.start()
	grpc_server.wait_for_termination()

def main(args):
	"""The DOCA gRPC Service.

	Args:
		args (list): list of command line arguments
	"""
	global logger, grpc_server_port

	parser = argparse.ArgumentParser(description=SERVICE_NAME)
	parser.add_argument('-d', '--debug',
	                    action='store_true',
	                    help='Set logging level to logging.DEBUG')

	args = parser.parse_args(args)
	if args.debug:
		logger.setLevel(logging.DEBUG)

	logger.info('Loading the configuration')

	try:
		load_config()
	except FileNotFoundError as e:
		logger.error(e)
		return 1
	except ValueError as e:
		logger.error(e)
		return 1

	server_threads = []
	for address, port in server_addresses:
		server_instance = threading.Thread(target=start_server_instance, args=(address, port))
		server_instance.start()
		server_threads.append(server_instance)

	# Wait for them all to complete
	for t in server_threads:
		t.join()

	return 0


if __name__ == '__main__':
	main(sys.argv[1:])
