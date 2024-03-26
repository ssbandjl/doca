/* SPDX-License-Identifier: BSD-3-Clause
 * Copyright(c) 2010-2015 Intel Corporation
 */

#ifndef _RTE_LHASH_H_
#define _RTE_LHASH_H_

/**
 * @file
 *
 * RTE List Hash Table
 */

#include "rte_hash.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Use the key directly as hash index. */
#define RTE_LHASH_DIRECT_KEY      (1u << 0)
/* stored data size does not exeed uintptr_t size */
#define RTE_LHASH_IMMEDIATE_DATA  (1u << 1)
/* List mostly used for append new. */
#define RTE_LHASH_WRITE_MOST      (1u << 2)

LIST_HEAD(rte_lhash_head, rte_lhash_entry);

struct rte_lhash {
	char name[RTE_HASH_NAMESIZE]; /**< Name of the hash list. */
	/**< number of buckets, need to be power of 2. */
	uint32_t buckets_num;
	uint32_t key_size;
	bool direct_key;
	struct rte_lhash_head *heads;	/**< list head arrays. */
};

struct rte_lhash_parameters {
	const char *name;
	uint32_t buckets_num;
	uint32_t entry_size; // const uint64_t
	uint32_t key_size; // hash key size in bytes
	uint64_t flags;
};

struct rte_lhash *rte_lhash_create(const struct rte_lhash_parameters *param);
int rte_lhash_add_key_data(const struct rte_lhash *h, const void * key, uint64_t data);

/**
 * @return
 *  0 if the key was located in a hash.
 * -ENOENT if the key was not located.
 */ 
int
rte_lhash_lookup(const struct rte_lhash *h, const void *key, uint64_t *data);
int rte_lhash_del_key(const struct rte_lhash *h, const void * key, uint64_t *data);
void rte_lhash_flush(struct rte_lhash *h);
int rte_lhash_free(struct rte_lhash *h);

#ifdef __cplusplus
}
#endif

#endif /* _RTE_LHASH_H_ */
