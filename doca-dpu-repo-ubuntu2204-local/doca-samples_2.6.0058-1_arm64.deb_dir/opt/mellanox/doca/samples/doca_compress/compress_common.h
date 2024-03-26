/*
 * Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES, ALL RIGHTS RESERVED.
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

#ifndef COMPRESS_COMMON_H_
#define COMPRESS_COMMON_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <endian.h>

#include <doca_dev.h>
#include <doca_compress.h>
#include <doca_mmap.h>
#include <doca_error.h>

#define USER_MAX_FILE_NAME 255			/* Max file name length */
#define MAX_FILE_NAME (USER_MAX_FILE_NAME + 1)	/* Max file name string length */
#define SLEEP_IN_NANOS (10 * 1000)		/* Sample the task every 10 microseconds */
#define NUM_COMPRESS_TASKS (1)			/* Number of compress tasks */
#define ADLER_CHECKSUM_SHIFT (32)		/* The shift for the Adler checksum within the output checksum */
#define ZLIB_HEADER_SIZE (2)			/* The header size in zlib */
#define ZLIB_TRAILER_SIZE (4)			/* The trailer size in zlib (the 32-bit checksum) */
/* Additional memory for zlib compatibility, used in allocation and reading */
#define ZLIB_COMPATIBILITY_ADDITIONAL_MEMORY (ZLIB_HEADER_SIZE + ZLIB_TRAILER_SIZE)

/* Compress modes */
enum compress_mode {
	COMPRESS_MODE_COMPRESS_DEFLATE,			/* Compress mode */
	COMPRESS_MODE_DECOMPRESS_DEFLATE,		/* Decompress mode */
};

/* Configuration struct */
struct compress_cfg {
	char file_path[MAX_FILE_NAME];			/* File to compress/decompress */
	char output_path[MAX_FILE_NAME];		/* Output file */
	char pci_address[DOCA_DEVINFO_PCI_ADDR_SIZE];	/* Device PCI address */
	enum compress_mode mode;			/* Compress task type */
	bool zlib_compatible;				/* Write \ read a file compatible with default zlib settings */
	bool output_checksum;				/* To output checksum or not */
};

/* DOCA compress resources */
struct compress_resources {
	struct program_core_objects *state;		/* DOCA program core objects */
	struct doca_compress *compress;			/* DOCA compress context */
	size_t num_remaining_tasks;			/* Number of remaining compress tasks */
	enum compress_mode mode;			/* Compress mode - compress/decompress */
	bool run_main_loop;				/* Controls whether progress loop should be run */
};

/* A zlib header, for comptability with third-party applications */
struct compress_zlib_header {
	uint8_t cmf;	/**< The CMF byte represents the Compression Method and Compression Info */
	uint8_t flg;	/**< The FLG byte represents various flags, including FCHECK and FLEVEL */
} __attribute__((packed));

/*
 * Register the command line parameters for the sample
 *
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t register_compress_params(void);

/*
 * Initiate the fields of the zlib header with default values
 *
 * @zlib_header [in]: A Zlib header to initiate with DOCA Compress default settings
 */
void init_compress_zlib_header(struct compress_zlib_header *zlib_header);

/*
 * Verify the header values are valid and compatible with DOCA compress
 *
 * @zlib_header [in]: A Zlib header to initiate with DOCA Compress default settings
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t verify_compress_zlib_header(struct compress_zlib_header *zlib_header);


/*
 * Allocate DOCA compress resources
 *
 * @pci_addr [in]: Device PCI address
 * @max_bufs [in]: Maximum number of buffers for DOCA Inventory
 * @resources [out]: DOCA compress resources to allocate
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t allocate_compress_resources(const char *pci_addr, uint32_t max_bufs,
					struct compress_resources *resources);

/*
 * Destroy DOCA compress resources
 *
 * @resources [in]: DOCA compress resources to destroy
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t destroy_compress_resources(struct compress_resources *resources);

/*
 * Submit compress deflate task and wait for completion
 * Also calculate the checksum where the lower 32 bits contain the CRC checksum result
 * and the upper 32 bits contain the Adler checksum result.
 *
 * @resources [in]: DOCA compress resources
 * @src_buf [in]: Source buffer
 * @dst_buf [in]: Destination buffer
 * @output_checksum [out]: The calculated checksum
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t submit_compress_deflate_task(struct compress_resources *resources, struct doca_buf *src_buf,
						struct doca_buf *dst_buf, uint64_t *output_checksum);

/*
 * Submit decompress deflate task and wait for completion
 * Also calculate the checksum where the lower 32 bits contain the CRC checksum result
 * and the upper 32 bits contain the Adler checksum result.
 *
 * @resources [in]: DOCA compress resources
 * @src_buf [in]: Source buffer
 * @dst_buf [in]: Destination buffer
 * @output_checksum [out]: The calculated checksum
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t submit_decompress_deflate_task(struct compress_resources *resources, struct doca_buf *src_buf,
						struct doca_buf *dst_buf, uint64_t *output_checksum);

/*
 * Check if given device is capable of executing a DOCA compress deflate task.
 *
 * @devinfo [in]: The DOCA device information
 * @return: DOCA_SUCCESS if the device supports DOCA compress deflate task and DOCA_ERROR otherwise
 */
doca_error_t compress_task_compress_is_supported(struct doca_devinfo *devinfo);

/*
 * Check if given device is capable of executing a DOCA decompress deflate task.
 *
 * @devinfo [in]: The DOCA device information
 * @return: DOCA_SUCCESS if the device supports DOCA decompress deflate task and DOCA_ERROR otherwise
 */
doca_error_t compress_task_decompress_is_supported(struct doca_devinfo *devinfo);

/*
 * Compress task completed callback
 *
 * @compress_task [in]: Completed task
 * @task_user_data [in]: doca_data from the task
 * @ctx_user_data [in]: doca_data from the context
 */
void compress_completed_callback(struct doca_compress_task_compress_deflate *compress_task,
					union doca_data task_user_data, union doca_data ctx_user_data);

/*
 * Compress task error callback
 *
 * @compress_task [in]: failed task
 * @task_user_data [in]: doca_data from the task
 * @ctx_user_data [in]: doca_data from the context
 */
void compress_error_callback(struct doca_compress_task_compress_deflate *compress_task,
					union doca_data task_user_data, union doca_data ctx_user_data);

/*
 * Decompress task completed callback
 *
 * @decompress_task [in]: Completed task
 * @task_user_data [in]: doca_data from the task
 * @ctx_user_data [in]: doca_data from the context
 */
void decompress_completed_callback(struct doca_compress_task_decompress_deflate *decompress_task,
					union doca_data task_user_data, union doca_data ctx_user_data);

/*
 * Decompress task error callback
 *
 * @decompress_task [in]: failed task
 * @task_user_data [in]: doca_data from the task
 * @ctx_user_data [in]: doca_data from the context
 */
void decompress_error_callback(struct doca_compress_task_decompress_deflate *decompress_task,
				union doca_data task_user_data, union doca_data ctx_user_data);

#endif /* COMPRESS_COMMON_H_ */
