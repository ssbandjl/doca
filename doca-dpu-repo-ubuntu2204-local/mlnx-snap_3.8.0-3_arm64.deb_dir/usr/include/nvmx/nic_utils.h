#ifndef NVMX_SRC_NIC_UTILS_H_
#define NVMX_SRC_NIC_UTILS_H_

typedef struct nic_utils nic_utils_t;

nic_utils_t* nic_utils_open();
void nic_utils_close(nic_utils_t *ctx);
char *nic_utils_exec(nic_utils_t *ctx, const char *func_name, const char *proto, ...);

#endif
