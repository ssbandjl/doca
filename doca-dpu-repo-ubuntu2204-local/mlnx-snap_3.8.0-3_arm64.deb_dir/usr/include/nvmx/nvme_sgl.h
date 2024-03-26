#ifndef _NVME_SGL_H
#define _NVME_SGL_H

#include "nvme.h"

#define SGL_SEGMENT_MAX_ITEMS_COUNT 16
#define SGL_SEGMENT_SIZE (SGL_SEGMENT_MAX_ITEMS_COUNT * sizeof(nvme_sgl_desc_t))

enum {
    SGL_DIR_WRITE, 
    SGL_DIR_READ
};

static inline bool nvme_sgl_needs_aux(nvme_async_req_t *req)
{
    bool ret;

    switch (req->sgl.sgl.type) {
    case NVME_SGL_TYPE_DATA_BLOCK:
        ret = le32_to_cpu(req->sgl.sgl.data_block.length) < req->len;
        break;
    case NVME_SGL_TYPE_KEYED_DATA_BLOCK:
        ret = le32_to_cpu(req->sgl.sgl.keyed_data_block.length) < req->len;
        break;
    default:
        ret = true;
    }

    return ret;
}

/* 
 * Read/write to/from sgl.
 * The length of data must not the maximal io size
 */
void nvme_sgl_rw(nvme_queue_ops_t *ops, const nvme_sgl_desc_t *sgl, void *buf,
                 size_t len, int dir);
int nvme_sgl_rw_nb(nvme_async_req_t *req);
void nvme_sgl_rw_iov_nb(nvme_async_req_t *req);
void nvme_sgl_rw_iov_data_nb(nvme_async_req_t *req);

/* write data to sgl */
static inline void
nvme_sgl_write(nvme_queue_ops_t *ops, const nvme_sgl_desc_t *sgl, void *buf,
               size_t len)
{
    nvme_sgl_rw(ops, sgl, buf, len, SGL_DIR_WRITE);
}

/* read data from sgl */
static inline void
nvme_sgl_read(nvme_queue_ops_t *ops, const nvme_sgl_desc_t *sgl, void *buf,
              size_t len)
{
    nvme_sgl_rw(ops, sgl, buf, len, SGL_DIR_READ);
}

static inline int
nvme_sgl_write_nb(nvme_async_req_t *req, const nvme_sgl_desc_t *sgl, size_t len)
{
    req->dma_cmd.op = NVME_REQ_DMA_TO_HOST;
    memcpy(&req->sgl.sgl, sgl, sizeof(nvme_sgl_desc_t));
    req->len = len;
    req->is_prp = false;

    return nvme_sgl_rw_nb(req);
}

static inline int
nvme_sgl_read_nb(nvme_async_req_t *req, const nvme_sgl_desc_t *sgl, size_t len)
{
    req->dma_cmd.op = NVME_REQ_DMA_FROM_HOST;
    memcpy(&req->sgl.sgl, sgl, sizeof(nvme_sgl_desc_t));
    req->len = len;
    req->is_prp = false;

    return nvme_sgl_rw_nb(req);
}
#endif
