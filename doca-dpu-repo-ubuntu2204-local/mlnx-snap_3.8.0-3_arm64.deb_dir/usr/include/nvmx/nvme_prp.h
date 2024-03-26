#ifndef _NVME_PRP_H
#define _NVME_PRP_H

/* PRP list handling */

#define PRP_PAGE_SIZE(pbits) (1ULL<<(pbits))
#define PRP_PAGE_BASE(prp, pbits) ((prp) & ~(PRP_PAGE_SIZE(pbits) - 1))
#define PRP_PAGE_OFFSET(prp, pbits) ((prp) & (PRP_PAGE_SIZE(pbits) - 1))

enum {
    PRP_DIR_WRITE, 
    PRP_DIR_READ
};

static inline bool nvme_prp_needs_aux(nvme_async_req_t *req)
{
    if (req->dma_cmd.op == NVME_REQ_DMA_FROM_HOST)
        return false;

    const size_t to_read = PRP_PAGE_SIZE(req->ops->page_bits) -
                           PRP_PAGE_OFFSET(req->prp.prp1, req->ops->page_bits);

    if (to_read >= req->len)
        return false;

    if (req->len - to_read <= PRP_PAGE_SIZE(req->ops->page_bits))
        return false;

    return true;
}

/* Read/write short prp list.
 * The length of data must not exceed two PRP pages
 */
enum nvme_status_code
nvme_prp_rw_short(nvme_queue_ops_t *ops, uint64_t prp1, uint64_t prp2, void *buf,
                  size_t len, int dir);

/* write data to prp */
static inline enum nvme_status_code
nvme_prp_write_short(nvme_queue_ops_t *ops, uint64_t prp1, uint64_t prp2, void *buf, size_t len)
{
    return nvme_prp_rw_short(ops, prp1, prp2, buf, len, PRP_DIR_WRITE);
}

/* read data from prp */
static inline enum nvme_status_code
nvme_prp_read_short(nvme_queue_ops_t *ops, uint64_t prp1, uint64_t prp2, void *buf, size_t len)
{
    return nvme_prp_rw_short(ops, prp1, prp2, buf, len, PRP_DIR_READ);
}

/**
 * nvme_prp_rw() - Start read or write prp operation
 * @req: async request
 *
 * The function starts prp operation which is described by the @req.
 * The @req->comp_cb() will be called once the
 * request is completed.
 *
 * Return: NVMX_INPROGRESS or error code
 */
int nvme_prp_rw(nvme_async_req_t *req);

int nvme_prp_to_writev2v(nvme_async_req_t *req);

/**
 * nvme_prp_read() - Start prp read
 * @req:  async request
 * @prp1: prp1 physical memory address as specified by the NVMe spec
 * @prp2: prp2 physical memory address ad specified by the NVMe spec
 * @len:  data length
 *
 * The function reads host memory described by the @prp1 and @prp2
 * to @req->data_buf. The @req->comp_cb() will be called once the
 * request is completed.
 *
 * Return: NVMX_INPROGRESS or error code
 */
static inline int nvme_prp_read(nvme_async_req_t *req, uint64_t prp1, uint64_t prp2, size_t len)
{
    req->dma_cmd.op = NVME_REQ_DMA_FROM_HOST;
    req->prp.prp1 = prp1;
    req->prp.prp2 = prp2;
    req->len = len;
    req->is_prp = true;
    return nvme_prp_rw(req);
}

/**
 * nvme_prp_write() - Start prp write
 * @req:  async request
 * @prp1: prp1 physical memory address as specified by the NVMe spec
 * @prp2: prp2 physical memory address ad specified by the NVMe spec
 * @len:  data length
 *
 * The function writes @req->data_buf to the host memory described by
 * the @prp1 and @prp2. The @req->comp_cb() will be called once the
 * request is completed.
 *
 * Return: NVMX_INPROGRESS or error code
 */
static inline int nvme_prp_write(nvme_async_req_t *req, uint64_t prp1, uint64_t prp2, size_t len)
{
    req->dma_cmd.op = NVME_REQ_DMA_TO_HOST;
    req->prp.prp1 = prp1;
    req->prp.prp2 = prp2;
    req->len = len;
    req->is_prp = true;
    return nvme_prp_rw(req);
}

int nvme_prp_write_bcopy(nvme_async_req_t *req, uint64_t prp1, uint64_t prp2,
                                       void *buf, size_t len);

int nvme_prp_read_bcopy(nvme_async_req_t *req, uint64_t prp1, uint64_t prp2,
                                        void *buf, size_t len);

static inline size_t nvme_prp_list_size(nvme_async_req_t *req)
{
    return sizeof(uint64_t) * req->data_len /
           PRP_PAGE_SIZE(req->ops->page_bits);
}
#endif
