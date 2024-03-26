#ifndef _NVME_REGS_H
#define _NVME_REGS_H

#include "nvme.h"

/* length of area that contains command registers
 * excluding CQ/SQ and doorbels
 */
#define NVME_EMU_REGS_SIZE  0x50

/* NVME registers */
#define SNAP_NVME_REG_CAP    0x00     /* capabilities */
#define SNAP_NVME_REG_VS     0x08     /* version */
#define SNAP_NVME_REG_INTMS  0x0C     /* interrupt mask set */
#define SNAP_NVME_REG_INTMC  0x10     /* interrupt mask clear */
#define SNAP_NVME_REG_CC     0x14     /* Controller config */
#define SNAP_NVME_REG_CSTS   0x1C     /* Controller status */
#define SNAP_NVME_REG_NSSR   0x20     /* NVM subsystem reset */
#define SNAP_NVME_REG_AQA    0x24     /* Admin Queue Attrs */
#define SNAP_NVME_REG_ASQ    0x28     /* Admin Submission Queue Base Addr */
#define SNAP_NVME_REG_ACQ    0x30     /* Admin Completion Queue Base Addr */
/* Optional registers */
#define SNAP_NVME_REG_CMBLOC 0x38     /* Controller memory buffer location */
#define SNAP_NVME_REG_CMBSZ  0x3C     /* Controller memory buffer size */
#define SNAP_NVME_REG_BPINFO 0x40     /* Boot partition info */
#define SNAP_NVME_REG_BPRSEL 0x44     /* Boot partition read select */
#define SNAP_NVME_REG_BPMBL  0x48     /* Boot prtition memory buffer */
#define SNAP_NVME_REG_LAST   (-1) 

#define NVME_DB_BASE    0x1000   /* offset of SQ/CQ doorbells */

#define NVME_BIT(n)   (1u<<(n))

/* register indexes */
#define SNAP_NVME_REG_CAP_IDX    0
#define SNAP_NVME_REG_VS_IDX     1
#define SNAP_NVME_REG_INTMS_IDX  2
#define SNAP_NVME_REG_INTMC_IDX  3
#define SNAP_NVME_REG_CC_IDX     4
#define SNAP_NVME_REG_CSTS_IDX   5
#define SNAP_NVME_REG_NSSR_IDX   6
#define SNAP_NVME_REG_AQA_IDX    7
#define SNAP_NVME_REG_ASQ_IDX    8
#define SNAP_NVME_REG_ACQ_IDX    9
/* Optional registers */
#define SNAP_NVME_REG_CMBLOC_IDX 10
#define SNAP_NVME_REG_CMBSZ_IDX  11
#define SNAP_NVME_REG_BPINFO_IDX 12
#define SNAP_NVME_REG_BPRSEL_IDX 13
#define SNAP_NVME_REG_BPMBL_IDX  14

#define SNAP_NVME_REG_MAX_DUMP_FUNC_LEN   256

enum {
    SNAP_NVME_REG_RO   = NVME_BIT(0),    /* read only */
    SNAP_NVME_REG_RW   = NVME_BIT(1),    /* read/write */
    SNAP_NVME_REG_RW1S = NVME_BIT(2),    /* read/write 1 to set */
    SNAP_NVME_REG_RW1C = NVME_BIT(3)     /* read/write 1 to clear */
};

/* controller capabilities */
int nvme_reg_cap_mqes(uint64_t cap);
int nvme_reg_cap_cqr(uint64_t cap);
int nvme_reg_cap_ams(uint64_t cap);
int nvme_reg_cap_to(uint64_t cap);
int nvme_reg_cap_dstrd(uint64_t cap);
int nvme_reg_cap_nssrs(uint64_t cap);
int nvme_reg_cap_css(uint64_t cap);
int nvme_reg_cap_bps(uint64_t cap);
int nvme_reg_cap_mpsmin(uint64_t cap);
int nvme_reg_cap_mpsmax(uint64_t cap);

/* controller version */
int nvme_reg_vs_ter(uint32_t vs);
int nvme_reg_vs_mnr(uint32_t vs);
int nvme_reg_vs_mjr(uint32_t vs);

/* controller configuration */
int nvme_reg_cc_en(uint32_t cc);
int nvme_reg_cc_css(uint32_t cc);
int nvme_reg_cc_mps(uint32_t cc);
int nvme_reg_cc_ams(uint32_t cc);
int nvme_reg_cc_shn(uint32_t cc);
int nvme_reg_cc_iosqes(uint32_t cc);
int nvme_reg_cc_iocqes(uint32_t cc);

/* controller status */
int nvme_reg_csts_rdy(uint32_t csts);
int nvme_reg_csts_cfs(uint32_t csts);
int nvme_reg_csts_shst(uint32_t csts);
int nvme_reg_csts_nssro(uint32_t csts);
int nvme_reg_csts_pp(uint32_t csts);

/* admin queue attrs */
int nvme_reg_aqa_asqs(uint32_t aqa);
int nvme_reg_aqa_acqs(uint32_t aqa);

typedef void (*nvme_reg_dump_func_t)(uint64_t reg, char *dump);

typedef struct nvme_register {
    unsigned              reg_base;
    unsigned              reg_size;
    uint8_t               reg_type;
    const char           *name;  
    const char           *desc; 
    nvme_reg_dump_func_t  reg_dump_func;
} nvme_register_t;

/**
 * I/O Command Set Selected
 *
 * Only a single command set is defined as of NVMe 1.3 (NVM).
 */
enum nvme_cc_css {
    NVME_CC_CSS_NVM = 0x0, /**< NVM command set */
};

#define NVME_CAP_CSS_NVM (1u << NVME_CC_CSS_NVM) /**< NVM command set supported */

union nvme_cc_register {
    uint32_t	raw;
    struct {
        /** enable */
        uint32_t en        : 1;
        uint32_t reserved1 : 3;
        /** i/o command set selected */
        uint32_t css       : 3;
        /** memory page size */
        uint32_t mps       : 4;
        /** arbitration mechanism selected */
        uint32_t ams       : 3;
        /** shutdown notification */
        uint32_t shn       : 2;
        /** i/o submission queue entry size */
        uint32_t iosqes    : 4;
        /** i/o completion queue entry size */
        uint32_t iocqes    : 4;
        uint32_t reserved2 : 8;
    } bits;
};

typedef struct nvme_bar {
    uint64_t    cap;
    uint32_t    vs;
    uint32_t    intms;
    uint32_t    intmc;
    uint32_t    cc;
    uint32_t    rsvd1;
    uint32_t    csts;
    uint32_t    nssrc;
    uint32_t    aqa;
    uint64_t    asq;
    uint64_t    acq;
    uint32_t    cmbloc;
    uint32_t    cmbsz;
    uint32_t    bpinfo;
    uint32_t    bprsel;
    uint32_t    bpmbl;
} nvme_bar_t;

typedef struct nvme_bar_instance {
    nvme_bar_t    curr;
    nvme_bar_t    prev;
    void          *ucontext;
} nvme_bar_instance_t;

typedef int (*nvme_bar_read_func_t)(void *ucontext, void *buf, uint32_t addr, unsigned len);
typedef int (*nvme_bar_write_func_t)(void *ucontext, void *buf, uint32_t addr, unsigned len);

/* called when register is modified */
typedef void (*nvme_reg_mod_cb_func_t)(void *bar, nvme_register_t *reg, uint64_t val,
				       uint64_t prev_val);

/* initialize bar and read cap & vs regs */
int nvme_bar_init(nvme_bar_read_func_t bar_reader, nvme_bar_instance_t *bar,
                  void *ucontext);
int nvme_bar_init_modify(nvme_bar_write_func_t bar_writer,
                         nvme_bar_instance_t *bar, void *ucontext);

/* update whole bar
 * Calls callback for each modified register
 */
int nvme_bar_update(nvme_bar_instance_t *bar, nvme_bar_read_func_t bar_reader,
		    nvme_reg_mod_cb_func_t cb);

/* dump a whole bar */
void nvme_bar_dump(void *bar, unsigned len);
/* dump register with given index */
void nvme_reg_dump(nvme_register_t *reg, void *bar, bool user_mode);

uint64_t nvme_reg_get(nvme_register_t *reg, void *bar);
void nvme_reg_set(nvme_register_t *reg, void *bar, uint64_t val);

extern nvme_bar_t nvme_bar_default_vals;
extern nvme_register_t nvme_regs[];
#endif

