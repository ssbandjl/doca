#ifndef _NVMF_H
#define _NVMF_H

#include <stdint.h>
#include <stdbool.h>

#include "nvme.h"
#include "compiler.h"

/* Minimum number of admin queue entries defined by NVMe over Fabrics spec */
#define NVMF_MIN_ADMIN_QUEUE_ENTRIES 32
#define NVMF_MAX_IO_QUEUE_SIZE 1024
#define NVMF_MIN_KA_TIMEOUT_MS 100

#define NVMF_TRADDR_MAX_LEN 256
#define NVMF_TRSVCID_MAX_LEN 32
#define NVMF_NQN_MAX_LEN 223

#define NVMF_DISCOVERY_NQN "nqn.2014-08.org.nvmexpress.discovery"
#define NVMF_NQN_UUID_PRE "nqn.2014-08.org.nvmexpress:uuid:"

/* Fabric Command Set */
#define NVME_OPCODE_FABRIC 0x7f

#define NVMF_PROP_SIZE_4  0
#define NVMF_PROP_SIZE_8  1

enum nvmf_fabric_cmd_types {
	NVMF_FABRIC_COMMAND_PROPERTY_SET		= 0x00,
	NVMF_FABRIC_COMMAND_CONNECT			= 0x01,
	NVMF_FABRIC_COMMAND_PROPERTY_GET		= 0x04,
	NVMF_FABRIC_COMMAND_AUTHENTICATION_SEND		= 0x05,
	NVMF_FABRIC_COMMAND_AUTHENTICATION_RECV		= 0x06,
};

enum nvmf_fabric_cmd_status_code {
	NVMF_FABRIC_SC_INCOMPATIBLE_FORMAT		= 0x80,
	NVMF_FABRIC_SC_CONTROLLER_BUSY			= 0x81,
	NVMF_FABRIC_SC_INVALID_PARAM			= 0x82,
	NVMF_FABRIC_SC_RESTART_DISCOVERY		= 0x83,
	NVMF_FABRIC_SC_INVALID_HOST			= 0x84,
	NVMF_FABRIC_SC_LOG_RESTART_DISCOVERY		= 0x90,
	NVMF_FABRIC_SC_AUTH_REQUIRED			= 0x91,
};

/*
 * NVM subsystem types
 */
enum nvmf_subsystem_type {
	/* Discovery type for NVM subsystem */
	NVMF_SUBTYPE_DISCOVERY	= 0x1,
	/* NVMe type for NVM subsystem */
	NVMF_SUBTYPE_NVME	= 0x2,
};

/*
 * NVMe over Fabrics transport types
 */
enum nvmf_trtype {
	/* RDMA */
	NVMF_TRTYPE_RDMA	= 0x1,
	/* Fibre Channel */
	NVMF_TRTYPE_FC		= 0x2,
};

/*
 * Address family types
 */
enum nvmf_adrfam {
	/* IPv4 (AF_INET) */
	NVMF_ADRFAM_IPV4	= 0x1,
	/* IPv6 (AF_INET6) */
	NVMF_ADRFAM_IPV6	= 0x2,
	/* InfiniBand (AF_IB) */
	NVMF_ADRFAM_IB		= 0x3,
	/* Fibre Channel */
	NVMF_ADRFAM_FC		= 0x4,
};

typedef struct NVME_PACKED nvmf_capsule_cmd {
	uint8_t			opcode;
	uint8_t			reserved1;
	uint16_t		cid;
	uint8_t			fctype;
	uint8_t			reserved2[35];
	uint8_t			fabric_specific[24];
} nvmf_capsule_cmd;

typedef struct NVME_PACKED nvmf_fabric_auth_recv_cmd {
	uint8_t			opcode;
	uint8_t			reserved1;
	uint16_t		cid;
	uint8_t			fctype;
	uint8_t			reserved2[19];
	union nvme_data_ptr	dptr;
	uint8_t			reserved3;
	uint8_t			spsp0;
	uint8_t			spsp1;
	uint8_t			secp;
	uint32_t		al;
	uint8_t			reserved4[16];
} nvmf_fabric_auth_recv_cmd;

typedef struct NVME_PACKED nvmf_fabric_auth_send_cmd {
	uint8_t			opcode;
	uint8_t			reserved1;
	uint16_t		cid;
	uint8_t			fctype;
	uint8_t			reserved2[19];
	union nvme_data_ptr	dptr;
	uint8_t			reserved3;
	uint8_t			spsp0;
	uint8_t			spsp1;
	uint8_t			secp;
	uint32_t		tl;
	uint8_t			reserved4[16];
} nvmf_fabric_auth_send_cmd;

typedef struct NVME_PACKED nvmf_fabric_connect_data {
	uint8_t			hostid[16];
	uint16_t		cntlid;
	uint8_t			reserved5[238];
	uint8_t			subnqn[256];
	uint8_t			hostnqn[256];
	uint8_t			reserved6[256];
} nvmf_fabric_connect_data;

typedef struct NVME_PACKED nvmf_fabric_connect_cmd {
	uint8_t			opcode;
	uint8_t			reserved1;
	uint16_t		cid;
	uint8_t			fctype;
	uint8_t			reserved2[19];
	union nvme_data_ptr	dptr;
	uint16_t		recfmt;
	uint16_t		qid;
	uint16_t		sqsize;
	uint8_t			cattr;
	uint8_t			reserved3;
	uint32_t		kato; /* keep alive timeout */
	uint8_t			reserved4[12];
} nvmf_fabric_connect_cmd;

typedef struct NVME_PACKED nvmf_fabric_connect_rsp {
	union {
		struct {
			uint16_t cntlid;
			uint16_t authreq;
		} success;

		struct {
			uint16_t	ipo;
			uint8_t		iattr;
			uint8_t		reserved;
		} invalid;

		uint32_t raw;
	} status_code_specific;

	uint32_t	reserved0;
	uint16_t	sqhd;
	uint16_t	reserved1;
	uint16_t	cid;
	uint16_t	status;
} nvmf_fabric_connect_rsp;

typedef struct NVME_PACKED nvmf_fabric_prop_get_cmd {
	uint8_t			opcode;
	uint8_t			reserved1;
	uint16_t		cid;
	uint8_t			fctype;
	uint8_t			reserved2[35];
	struct {
		uint8_t size		: 2;
		uint8_t reserved	: 6;
	} attrib;
	uint8_t			reserved3[3];
	uint32_t		ofst;
	uint8_t			reserved4[16];
} nvmf_fabric_prop_get_cmd;

typedef struct NVME_PACKED nvmf_fabric_prop_get_rsp {
	union {
		uint64_t u64;
		struct {
			uint32_t low;
			uint32_t high;
		} u32;
	} value;

	uint16_t	sqhd;
	uint16_t	reserved0;
	uint16_t	cid;
	uint16_t	status;
} nvmf_fabric_prop_get_rsp;


typedef struct NVME_PACKED nvmf_fabric_prop_set_cmd {
	uint8_t			opcode;
	uint8_t			reserved1;
	uint16_t		cid;
	uint8_t			fctype;
	uint8_t			reserved2[35];
	struct {
		uint8_t size		: 2;
		uint8_t reserved	: 6;
	} attrib;
	uint8_t			reserved3[3];
	uint32_t		ofst;
	uint64_t		value;
	uint8_t			reserved4[8];
} nvmf_fabric_prop_set_cmd;

/*
 * NVMe-oF transport identifier.
 * This identifies a unique endpoint on an NVMe-oF fabric.
 */
struct nvmf_transport_id {
	enum nvmf_trtype trtype;
	enum nvmf_adrfam adrfam;
	/*
	 * Transport address of the NVMe-oF endpoint. For transports which use
	 * IP addressing (e.g. RDMA), this should be an IP address.
	 */
	char traddr[NVMF_TRADDR_MAX_LEN + 1];
	/*
	 * Transport service id of the NVMe-oF endpoint. For transports which
	 * use IP addressing (e.g. RDMA), this field should be the port number.
	 */
	char trsvcid[NVMF_TRSVCID_MAX_LEN + 1];
	/*
	 * Subsystem NQN of the NVMe over Fabrics endpoint. May be a zero length
	 * string.
	 */
	char subnqn[NVMF_NQN_MAX_LEN + 1];
};

typedef struct nvmf_completion_status {
    nvme_cqe_t cqe;
    bool done;
} nvmf_completion_status_t;

#endif
