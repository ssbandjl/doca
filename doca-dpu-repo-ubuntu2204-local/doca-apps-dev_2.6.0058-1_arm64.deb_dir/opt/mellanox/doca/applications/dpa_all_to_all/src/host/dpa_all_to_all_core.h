/*
 * Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES, ALL RIGHTS RESERVED.
 *
 * This software product is a proprietary product of NVIDIA CORPORATION &
 * AFFILIATES (the "Company") and all right, title, and interest in and to the
 * software product, including all associated intellectual property rights, are
 * and shall remain exclusively with the Company.
 *
 * This software product is governed by the End User License Agreement
 * provided with the software product.
 *
 */

#ifndef DPA_ALL_TO_ALL_CORE_H_
#define DPA_ALL_TO_ALL_CORE_H_

#include <mpi.h>

#include <doca_dpa.h>
#include <doca_buf_array.h>
#include <doca_mmap.h>
#include <doca_dev.h>
#include <doca_error.h>
#include <doca_sync_event.h>
#include <doca_rdma.h>

#define MAX_DEVICES (2)							/* Max number of IB devices to use*/
#define MAX_USER_IB_DEVICE_NAME_LEN (256)				/* Maximum user IB device name string length */
#define MAX_IB_DEVICE_NAME_LEN (MAX_USER_IB_DEVICE_NAME_LEN + 1)	/* Maximum IB device name string length */
#define IB_DEVICE_DEFAULT_NAME "NOT_SET"				/* IB device default name */
#define MAX_NUM_THREADS (8)						/* Maximum number of threads to run the kernel */
#define MESSAGE_SIZE_DEFAULT_LEN (-1)					/* Message size default length */
#define MAX_NUM_PROC (16)						/* Maximum number of processes */
#define SYNC_EVENT_MASK_FFS (0xFFFFFFFFFFFFFFFF)			/* Mask for doca_sync_event_wait_gt() wait value */

/* Configuration struct */
struct a2a_config {
	int msgsize;						/* Message size of sendbuf (in bytes) */
	char device1_name[MAX_IB_DEVICE_NAME_LEN];		/* Buffer that holds the IB device name */
	char device2_name[MAX_IB_DEVICE_NAME_LEN];		/* Buffer that holds the IB device name */
};

/* A struct that includes all the resources needed for DPA */
struct a2a_resources {
	char device_name[MAX_IB_DEVICE_NAME_LEN];				/* Buffer that holds the IB device name */
	struct doca_dev *doca_device;						/* DOCA device for DPA */
	struct doca_dpa *doca_dpa;						/* DOCA DPA context */
	void *sendbuf;								/* The send buffer we get from the alltoall call */
	void *recvbuf;								/* The receive buffer we get from the alltoall call */
	struct doca_sync_event *comp_event;					/* DOCA sync event for DPA completion event */
	uint64_t a2a_seq_num;							/* Sequence number for the completion event */
	struct doca_sync_event **kernel_events;					/* DOCA sync events for kernel */
	doca_dpa_dev_sync_event_t *kernel_events_handle;			/* DOCA sync events handles for DPA kernel */
	doca_dpa_dev_uintptr_t devptr_kernel_events_handle;			/* DOCA DPA local processes remote events for kernel device pointers */
	uint8_t *lp_remote_kernel_events_export_data;				/* DOCA DPA local process device remote events export data */
	size_t *lp_remote_kernel_events_export_sizes;				/* DOCA DPA local process device remote events export sizes */
	const uint8_t **rp_remote_kernel_events_export_data;			/* DOCA DPA remote process device remote events export data */
	size_t *rp_remote_kernel_events_export_sizes;				/* DOCA DPA remote process device remote events export sizes */
	struct doca_sync_event_remote_net **rp_kernel_events;			/* DOCA DPA remote process device remote events */
	doca_dpa_dev_sync_event_remote_net_t *rp_kernel_events_dpa_handles;	/* DOCA DPA remote process device remote events */
	doca_dpa_dev_uintptr_t devptr_rp_remote_kernel_events;			/* DOCA DPA remote processes remote events DPA handles */
	struct doca_rdma **rdmas;						/* DOCA RDMA contexts */
	doca_dpa_dev_uintptr_t devptr_rdmas;					/* DOCA DPA RDMA handlers device pointers */
	struct doca_mmap *sendbuf_mmap;						/* DOCA mmap for sendbuf host memory */
	struct doca_mmap *recvbuf_mmap;						/* DOCA mmap for recvbuf host memory */
	struct doca_buf_arr *sendbuf_buf_arr;					/* DOCA buf array for sendbuf */
	struct doca_dpa_dev_buf_arr *sendbuf_dpa_buf_arr;			/* DOCA buf array DPA handle for sendbuf */
	struct doca_mmap **export_mmaps;					/* DOCA mmap export from remote processes */
	struct doca_buf_arr **from_export_buf_arrs;				/* DOCA buf arrays from exported mmaps */
	struct doca_dpa_dev_buf_arr **from_export_dpa_buf_arrs;			/* DOCA buf array DPA handles from exported mmaps */
	doca_dpa_dev_uintptr_t devptr_recvbufs_buf_arr_handles;			/* DOCA DPA recvbuf buf array handles device pointers */
	int num_ranks;								/* Number of running processes */
	int my_rank;								/* Rank of the current process */
	int mesg_count;								/* Message count */
	MPI_Datatype msg_type;							/* MPI Datatype of the message */
	MPI_Aint extent;							/* The extent of the message type */
	MPI_Comm comm;								/* MPI communication group */
};

/* DPA Alltoall request that is used to check the completion of the non-blocking alltoall call */
struct dpa_a2a_request {
	struct a2a_resources *resources;			/* Alltoall resources */
};

/*
 * Check if the provided device name is a name of a valid IB device with DPA capabilities
 *
 * @device_name [in]: The wanted IB device name with DPA capabilites
 * @return: True if device_name is IB_DEVICE_DEFAULT_NAME or if an IB device with DPA capabilities with name
 *	same as device_name is found, false otherwise.
 */
bool dpa_device_exists_check(const char *device_name);

/*
 * Finalize the request and destroy the associated resources
 *
 * @req [in]: DOCA DPA alltoall request that was sent to a previous ialltoall call
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t dpa_a2a_req_finalize(struct dpa_a2a_request *req);

/*
 * Wait for the requested non-blocking alltoall call to finish
 *
 * @req [in]: DPA alltoall request
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t dpa_a2a_req_wait(struct dpa_a2a_request *req);

/*
 * Initialize the DOCA DPA all to all resources, which includes creating DOCA DPA context, allocating and connecting
 * DOCA DPA endpoints and creating DOCA DPA host and device memories.
 *
 * @resources [in/out]: All to all resources
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t dpa_a2a_init(struct a2a_resources *resources);

/*
 * Destroy the all to all resources
 *
 * @resources [in]: All to all resources
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t dpa_a2a_destroy(struct a2a_resources *resources);

/*
 * MPI non-blocking all to all using DOCA DPA. This function sends data to all processes from all processes.
 *
 * @sendbuf [in]: The starting address of send buffer
 * @sendcount [in]: The number of elements to be sent to each process
 * @sendtype [in]: The datatype of the receive buff elements
 * @recvbuf [in]: The starting address of the receive buffer
 * @recvcount [in]: The number of elements to be received from each process
 * @recvtype [in]: The datatype of the send buff elements
 * @comm [in]: The communicator over which the data is to be exchanged
 * @req [out]: DPA request that is used to check if the alltoall is finished using dpa_mpi_req_wait().
 *		Note that after finishing we must finalize the request using dpa_mpi_req_finalize().
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t dpa_ialltoall(void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount,
				MPI_Datatype recvtype, MPI_Comm comm, struct dpa_a2a_request *req);

/*
 * MPI blocking all to all using DOCA DPA. This function sends data to all processes from all processes.
 *
 * @sendbuf [in]: The starting address of send buffer
 * @sendcount [in]: The number of elements to be sent to each process
 * @sendtype [in]: The datatype of the receive buff elements
 * @recvbuf [in]: The starting address of the receive buffer
 * @recvcount [in]: The number of elements to be received from each process
 * @recvtype [in]: The datatype of the send buff elements
 * @comm [in]: The communicator over which the data is to be exchanged
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t dpa_alltoall(void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount,
				MPI_Datatype recvtype, MPI_Comm comm);

/*
 * Perform all to all example using DOCA DPA
 *
 * @argc [in]: command line arguments size
 * @argv [in]: array of command line arguments
 * @cfg [in]: All to all user configurations
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t dpa_a2a(int argc, char **argv, struct a2a_config *cfg);

#endif /* DPA_ALL_TO_ALL_CORE_H_ */
