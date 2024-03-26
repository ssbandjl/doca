#ifndef _SNAP_BLK_OPS_H
#define _SNAP_BLK_OPS_H

#include <sys/uio.h>
#include <stdbool.h>

/**
 * enum snap_bdev_op_status - Return status values for snap_bdev_io_done_cb_t
 * @SNAP_BDEV_OP_SUCCESS:	operation finished successfully
 * @SNAP_BDEV_OP_IO_ERROR:	operation failed due to IO error
 */
enum snap_bdev_op_status {
	SNAP_BDEV_OP_SUCCESS,
	SNAP_BDEV_OP_IO_ERROR,
};

struct ibv_mr;

/**
 * typedef snap_bdev_io_done_cb_t - callback on io operations done
 * @status:	status of the finished operation
 * @done_arg:	user context given on operation request
 *
 * callback function called by block device when operation is finished
 */
typedef void (*snap_bdev_io_done_cb_t)(enum snap_bdev_op_status status,
				       void *done_arg);

typedef void (*snap_mem_pool_ready_cb_t)(void *data, struct ibv_mr *mr,
					void *user);

/**
 * struct snap_bdev_io_done_ctx - context given for bdev ops
 * @cb:		callback on io operation done
 * @user_arg:	user opaque argument given to cb
 */
struct snap_bdev_io_done_ctx {
	snap_bdev_io_done_cb_t cb;
	void *user_arg;
};

struct snap_blk_mempool_ctx {
	void *ctx;
	void *user;
	snap_mem_pool_ready_cb_t callback;
	void *tag;
	int thread_id;
};

/**
 * struct snap_bdev_ops - operations provided by block device
 * @read:		pointer to function which reads blocks from bdev
 * @write:		pointer to function which writes blocks to bdev
 * @flush:		pointer to function which flushes bdev
 * @write_zeros:	pointer to function which writes zeros to bdev
 * @discard:		pointer to function which discards blocks in bdev
 * @dma_malloc:		pointer to function which allocates host memory
 *			(like malloc). some block device frameworks (e.g. spdk)
 *			require their own malloc-like function to be used.
 * @dma_free:		free memory allocated by dma_malloc
 * @get_num_blocks:	pointer to function which gets number of blocks in bdev
 * @get_block_size:	pointer to function which gets bdev block size
 * @get_bdev_name:	pointer to function which returns null terminated bdev
 *			name
 * @is_zcopy:		pointer to function which returns true if bdev supports
 *			ZCOPY
 * @is_zcopy_aligned:	pointer to function which returns true if address is
 *			ZCOPY and bdev aligned
 *
 * operations provided by the block device given to the virtio controller
 * ToDo: add mechanism to tell which block operations are supported
 */
struct snap_bdev_ops {
	int (*readv_blocks)(void *ctx, struct iovec *iov, int iovcnt,
		    uint64_t offset_blocks, uint64_t num_blocks,
		    struct snap_bdev_io_done_ctx *done_ctx, int thread_id);
	int (*writev_blocks)(void *ctx, struct iovec *iov, int iovcnt,
		     uint64_t offset_blocks, uint64_t num_blocks,
		     struct snap_bdev_io_done_ctx *done_ctx, int thread_id);
	int (*read)(void *ctx, void *buf, uint64_t offset,
			uint64_t len, struct snap_bdev_io_done_ctx *done_ctx,
			int thread_id);
	int (*write)(void *ctx, void *buf, uint64_t offset,
			uint64_t len, struct snap_bdev_io_done_ctx *done_ctx,
			int thread_id);
	int (*flush)(void *ctx, uint64_t offset_blocks, uint64_t num_blocks,
		     struct snap_bdev_io_done_ctx *done_ctx, int thread_id);
	int (*write_zeroes)(void *ctx,
			    uint64_t offset_blocks, uint64_t num_blocks,
			    struct snap_bdev_io_done_ctx *done_ctx,
			    int thread_id);
	int (*discard)(void *ctx, uint64_t offset_blocks, uint64_t num_blocks,
		       struct snap_bdev_io_done_ctx *done_ctx, int thread_id);
	void *(*dma_malloc)(size_t size);
	void (*dma_free)(void *buf);
	uint64_t (*get_num_blocks)(void *ctx);
	uint32_t (*get_block_size)(void *ctx);
	const char *(*get_bdev_name)(void *ctx);
	bool (*is_zcopy)(void *ctx);
	bool (*zcopy_validate_params)(void *ctx, struct iovec *iov,
				size_t iov_cnt,	uint64_t offset, uint64_t len);
	int (*dma_pool_malloc)(size_t size, struct snap_blk_mempool_ctx *mem_ctx);
	void (*dma_pool_cancel)(struct snap_blk_mempool_ctx *mem_ctx);
	void (*dma_pool_free)(struct snap_blk_mempool_ctx *ctx, void *buf);
	bool (*dma_pool_enabled)(void *ctx);
};

#endif
