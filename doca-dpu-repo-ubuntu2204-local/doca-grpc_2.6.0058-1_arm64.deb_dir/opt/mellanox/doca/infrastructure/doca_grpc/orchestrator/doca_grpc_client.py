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
import click
import logging
import shlex
import concurrent.futures as futures
import threading
import grpc

import common_pb2 as gen_common
import doca_grpc_orchestrator_pb2 as gen_pbuf
import doca_grpc_orchestrator_pb2_grpc as gen_grpc

BASE_NAME = 'DOCA gRPC Orchestrator'
CLIENT_NAME = BASE_NAME + ' Client'

gRPC_PORT = gen_common.eNetworkPort.k_DocaGrpcOrchestrator

logger = logging.getLogger(CLIENT_NAME)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(ch)
logger.setLevel(logging.INFO)


def get_program_list(stub):
	"""Command - Get the list of gRPC-supported programs

	Args:
		stub (grpc): grpc client
	"""
	logger.debug('CLI Command - Get Program List')
	program_list = stub.GetProgramList(gen_pbuf.ProgramListReq())
	logger.info(f'The list of gRPC supported programs on the DPU is:')
	logger.info(', '.join(program_list.program_names))


def spawn_program(stub, program_name, cmdline, port=None):
	"""Command - Spawn a gRPC-supported program

	Args:
		stub (grpc): grpc client
		program_name (str): program name, as returned from get_program_list()
		cmdline (str): cmdline for the spawned program
		port (int, optional): dedicated TCP port for the gRPC program (None by default)
	"""
	logger.debug('CLI Command - Spawn gRPC Program on the DPU')
	if port is not None:
		port = int(port)
	rich_status = stub.Create(gen_pbuf.Args(program_name=program_name, cmdline=cmdline, port=port))
	if rich_status.err_status.is_error:
		logger.error(rich_status.err_status.error_msg)
		return
	logger.info(f'Program UID is: {rich_status.uid.uid}')


def terminate_program(stub, program_uid):
	"""Command - Terminate a gRPC-supported program

	Args:
		stub (grpc): grpc client
		program_uid (str): program unique id: <ip address>:<port>
	"""
	logger.debug('CLI Command - Terminate a gRPC Program on the DPU')
	status = stub.Destroy(gen_pbuf.Uid(uid=program_uid))
	if status.is_error:
		logger.error(status.error_msg)
		return
	logger.info(f'Successfully terminated program {program_uid} on the DPU')


def connect_to_server(ctx):
	"""Connect to the gRPC Service on the DPU.

	Args:
		ctx (click ctx): click ctx holding the server address

	Return Value:
		DocaOrchestratorStub for the gRPC connection
	"""
	full_address = ctx.obj['full_address']
	logger.info(f'Connecting to the {BASE_NAME} Service on the DPU: {full_address}')
	channel = grpc.insecure_channel(f'{full_address}')

	return gen_grpc.DocaOrchestratorStub(channel)

@click.group(help='DOCA gRPC Client CLI tool')
@click.argument('server_address')
@click.option('-d', '--debug', is_flag=True, default=False)
@click.pass_context
def cli(ctx, server_address, debug):
	ctx.ensure_object(dict)

	if debug:
		logger.setLevel(logging.DEBUG)

	if ':' not in server_address:
		server_address = f'{server_address}:{gRPC_PORT}'

	ctx.obj['full_address'] = server_address

@cli.command()
@click.pass_context
def list(ctx):
	"""List the names of gRPC-supported program.
	"""
	get_program_list(connect_to_server(ctx))

@cli.command(context_settings=dict(
	allow_interspersed_args=False,
))
@click.argument('program_name')
@click.argument('program_args', nargs=-1, type=click.UNPROCESSED)
@click.option('-p', '--port', default=None, type=int,
	help='TCP Port to be used by the gRPC Program (If wishing to use a non-default port).')
@click.pass_context
def create(ctx, program_name, program_args, port):
	"""Create PROGRAM_NAME [PROGRAM_ARGS]...

	Spawn a program's gRPC-Server on the DPU, and pass it the command line arguments.
	"""
	spawn_program(connect_to_server(ctx), program_name, ' '.join(program_args), port=port)

@cli.command()
@click.argument('program_uid')
@click.pass_context
def destroy(ctx, program_uid):
	"""Destroy PROGRAM_UID

	Terminate the execution of the program matching the program UID.
	"""
	try:
		terminate_program(connect_to_server(ctx), program_uid)
	except grpc._channel._InactiveRpcError as e:
		logger.error('Failed to issue the "destroy" command, the gRPC service is probably busy')

if __name__ == '__main__':
	try:
		cli(obj={})
	except RuntimeError as e:
		logger.error('Failed to connect to the gRPC service on the DPU')
