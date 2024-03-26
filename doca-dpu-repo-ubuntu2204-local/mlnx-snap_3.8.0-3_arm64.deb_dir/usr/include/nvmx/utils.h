#ifndef _UTILS_H
#define _UTILS_H

#include <stdbool.h>
#include <net/if_arp.h>
#include "compiler.h"
#include "nvme.h"

#define IFACE_MAX_LEN member_size(struct arpreq, arp_dev)

int compare_uint32_t(const void *p, const void *q);
int iface_rgid_to_rmac(const char *dev, uint8_t *rgid, uint8_t *rmac);
int dev_to_iface(const char *dev, char *iface);
int ipaddr_to_iface(const char *addr, char *iface, size_t iface_sz);
int iface_to_mlx_dev(const char *iface, char *dev, size_t dev_sz);
int iface_to_ipaddr(const char *iface, char *ip_addr, size_t ip_addr_sz);
int rdma_dev_name_to_id(const char *rdma_dev_name);
bool is_global_id_equal(nvme_id_ns_descriptor_t *id_desc,
                        int nidt, nvme_id_ns_global_t *nid);
unsigned ulog2(uint32_t v);
unsigned char char_to_hex(unsigned char c);

#endif
