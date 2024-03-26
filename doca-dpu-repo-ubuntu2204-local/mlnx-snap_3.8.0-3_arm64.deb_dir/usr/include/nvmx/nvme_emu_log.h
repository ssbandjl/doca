#ifndef _NVME_LOG_H
#define _NVME_LOG_H

#include <stdio.h>
#include <stdlib.h>
#include "debug.h"

enum NvmxEmuLogLevel {
    NVMX_LOG_VERBOSE_LEVEL_MAX,

    NVMX_LOG_VERBOSE_LEVEL_DBG,
    NVMX_LOG_VERBOSE_LEVEL_INFO,
    NVMX_LOG_VERBOSE_LEVEL_WARN,
    NVMX_LOG_VERBOSE_LEVEL_ERR,
};

void nvmx_logger(const char *file_name, unsigned int line_num,
                enum NvmxEmuLogLevel level, const char *level_c,
                const char *format, ...);

int nvme_init_logger();
int nvme_close_logger();

#define __NVMX_LOG_COMMON(level, fmt, ...)                                  \
    do {                                                                    \
         nvmx_logger(__FILE__, __LINE__, NVMX_LOG_VERBOSE_LEVEL_##level,    \
         #level, fmt, ## __VA_ARGS__);                                              \
    } while (0);


#if NVMX_TRACE_DATA
#define nvmx_debug_data(_fmt, ...) \
    __NVMX_LOG_COMMON(DBG, _fmt, ## __VA_ARGS__)
#else
    #define nvmx_debug_data(_fmt, ...)
#endif

#ifdef DEBUG_ENABLED
#define nvmx_debug(_fmt, ...) \
    __NVMX_LOG_COMMON(DBG, _fmt, ## __VA_ARGS__)
#else
#define nvmx_debug(_fmt, ...)
#endif

#define nvmx_info(_fmt, ...) \
    __NVMX_LOG_COMMON(INFO, _fmt, ## __VA_ARGS__)

#define nvmx_warn(_fmt, ...) \
    __NVMX_LOG_COMMON(WARN, _fmt, ## __VA_ARGS__)

#define nvmx_error(_fmt, ...) \
    __NVMX_LOG_COMMON(ERR, _fmt, ## __VA_ARGS__)

#define nvmx_fatal(_fmt, ...) \
    do { \
        __NVMX_LOG_COMMON(ERR, _fmt, ## __VA_ARGS__) \
        nvmx_error_freeze(); \
        debug_print_backtrace(); \
        abort(); \
    } while(0);

#define nvmx_assertv_always(_expr, _fmt, ...) \
    do { \
        if (!(_expr)) { \
            nvmx_fatal("assertion failure: %s " _fmt, #_expr, ## __VA_ARGS__); \
        } \
    } while(0);


#endif
