#ifndef _SPDK_IO_MGR_H
#define _SPDK_IO_MGR_H
#include <stdio.h>
#include <stdint.h>

size_t spdk_io_mgr_get_num_threads();
struct spdk_thread *spdk_io_mgr_get_thread(int id);

int spdk_io_mgr_init();
void spdk_io_mgr_clear();
#endif
