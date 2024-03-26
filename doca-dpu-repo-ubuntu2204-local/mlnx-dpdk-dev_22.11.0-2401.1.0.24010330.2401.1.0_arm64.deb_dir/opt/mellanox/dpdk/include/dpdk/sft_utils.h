/* SPDX-License-Identifier: BSD-3-Clause
 * Copyright(c) 2010-2014 Intel Corporation
 */

#ifndef _SFT_UTILS_H_
#define _SFT_UTILS_H_

#include <stdint.h>

#include <rte_common.h>
#include <rte_lhash.h>

#include "rte_sft.h"
/**
 * @file
 * SFT utilities
 */

#define RTE_SFT_QUERY_FREQ_US 1000000

struct sft_mbuf {
	const struct rte_mbuf *m_in;
	struct rte_mbuf *m_out;
};

struct client_obj {
	LIST_ENTRY(client_obj) chain;
	const void *obj;
	uint8_t id;
};

/*
 * @direction_key structure is 12 bytes long
 * exactly as eth.dst, eth.src sequence
 */
struct direction_key {
	uint64_t k1;
	uint32_t k2;
} __rte_packed;

struct sft_lib_entry {
	TAILQ_ENTRY(sft_lib_entry) next;
	uint32_t fid;
	/* entry zone is required to find out if mbuf was sent from
	 * initiator or target connection side
	 */
	uint32_t zone;
	uint16_t queue;
	uint16_t l2_len;
	uint8_t app_state; /**< application defined flow state */
	uint8_t proto_state; /**< protocol state */
	uint8_t proto;
	uint8_t event_dev_id;
	uint8_t event_port_id;
	uint8_t ct_enable;
	uint32_t *data;
	uint64_t ns_rx_timestamp;
	/* initiator 7tuple determines direction of active mbuf.buf_addr
	 * alternative is to extract it live from mbuf and run a hash search
	 */
	struct rte_sft_7tuple stpl[2];
	struct rte_sft_entry *sft_entry[2];
	struct rte_sft_actions_specs action_specs;
	/* this is per queue list - no lock required */
	LIST_HEAD(, client_obj) client_objects_head;
	uint64_t nb_bytes_sw[2]; /**< Number of bytes passed in the sw flow. */
	uint64_t nb_bytes_hw[2]; /**< Number of bytes passed in the hw flow. */
	uint64_t nb_packets_sw[2]; /**< Number of packets passed in sw flow. */
	uint64_t nb_packets_hw[2]; /**< Number of packets passed in hw flow. */
	time_t last_activity_ts; /**< number of seconds since the Epoch */
	bool aged;
	bool offload;
	void *ct_obj;
	struct direction_key direction_key;
};

TAILQ_HEAD(sft_lib_entries, sft_lib_entry);

/* ID generation structure. */
struct sft_id_pool {
	uint32_t *free_arr; /**< Pointer to the a array of free values. */
	uint32_t base_index;
	/**< The next index that can be used without any free elements. */
	uint32_t *curr; /**< Pointer to the index to pop. */
	uint32_t *last; /**< Pointer to the last element in the empty arrray. */
	uint32_t max_id; /**< Maximum id can be allocated from the pool. */
};

struct sft_id_pool *sft_id_pool_alloc(uint32_t max_id);
void sft_id_pool_release(struct sft_id_pool *pool);
uint32_t sft_id_get(struct sft_id_pool *pool, uint32_t *id);
uint32_t sft_id_release(struct sft_id_pool *pool, uint32_t id);
void sft_query_alarm(void *param);
int sft_set_alarm(void);
int sft_cancel_alarm(void);

static inline void
sft_search_hash(const struct rte_lhash *h, const void *key, void **data)
{
	int ret = rte_lhash_lookup(h, key, (uint64_t *)data);

	if (ret == -ENOENT) {
		*data = NULL;
		ret = 0;
	}
};

static inline void
sft_reverse_5tuple(struct rte_sft_5tuple *dst, const struct rte_sft_5tuple *src)
{
	dst->is_ipv6 = src->is_ipv6;
	dst->proto = src->proto;
	dst->src_port = src->dst_port;
	dst->dst_port = src->src_port;
	if (!dst->is_ipv6) {
		dst->ipv4.src_addr = src->ipv4.dst_addr;
		dst->ipv4.dst_addr = src->ipv4.src_addr;
	} else {
		memcpy(dst->ipv6.src_addr, src->ipv6.dst_addr, 16);
		memcpy(dst->ipv6.dst_addr, src->ipv6.src_addr, 16);
	}
}

static inline bool
sft_match_directions(const struct sft_lib_entry *entry,
		     const struct rte_mbuf *m)
{
	const struct direction_key *mk = rte_pktmbuf_mtod(m, typeof(mk));

	return entry->direction_key.k1 == mk->k1
	       && entry->direction_key.k2 == mk->k2;
}

const struct rte_lhash *tcp_ct_hash(uint16_t queue);
int
sft_tcp_drain_mbuf(struct sft_lib_entry *entry,
		   const struct rte_mbuf **mbuf_out, uint16_t nb_out,
		   struct rte_sft_flow_status *status);
void
sft_tcp_track_conn(struct sft_mbuf *smb, struct rte_sft_mbuf_info *mif,
		   const struct sft_lib_entry *entry,
		   struct rte_sft_flow_status *statust,
		   struct rte_sft_error *error);
int
sft_tcp_start_track(struct sft_lib_entry *entry, struct rte_sft_error *error);

int
sft_tcp_stop_conn_track(const struct sft_lib_entry *entry,
			struct rte_sft_error *error);
const char *
sft_ct_state_name(enum sft_ct_state state);
#define SFT_PORT_ANY UINT16_MAX
/*
 * RFC0791 suggests to keep IP fragments up to 15 seconds
 */
#define SFT_IPFRAG_TIMEOUT 15
#endif /* _SFT_UTILS_H_*/

