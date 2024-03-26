/*
 * Copyright (c) 2022-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 */

#ifndef IBDIAGNET_CONTROL_API_H
#define IBDIAGNET_CONTROL_API_H

#include <sys/types.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CONTROL_API_VERSION 0x01000000

typedef void *control_session_handle_t;

/*
 * Control library should set all unsupported bits (unsupported stage or extra bits for supported
 * stage) to 1.
 */
typedef u_int64_t control_stage_flags_t[4];

typedef union
{
    struct
    {
        u_int64_t port_counters             : 1;
        u_int64_t port_counters_extended    : 1;
        u_int64_t extended_speeds           : 1;
        u_int64_t llr_statistics            : 1;
        u_int64_t port_rcv_error_details    : 1;
        u_int64_t port_xmit_discard_details : 1;
    } pm;
    control_stage_flags_t flags;
} control_pm_stage_flags_t;

/*
 * start_port and end_port must be valid port numbers for that node.
 */
typedef struct {
    u_int64_t node_guid;
    u_int16_t start_port;
    u_int16_t end_port;
} control_scope_record_t;

/*
 * If num_records = 0, then scope is full: all nodes and all ports.
 */
typedef struct {
    size_t                  num_records;
    control_scope_record_t *records;
} control_scope_t;

/*
 * Define CONTROL_IMPL to get function declarations instead of function pointers typedefs.
 * Control library shall define CONTROL_IMPL before including this file:
 *
 * #define CONTROL_IMPL
 * #include "ibdiagnet_control_api.h"
 */
#ifdef CONTROL_IMPL
    #define EXPORT_SYMBOL __attribute__((visibility("default")))

    EXPORT_SYMBOL u_int32_t control_get_api_version();

    EXPORT_SYMBOL control_session_handle_t control_open_session(u_int64_t);
    EXPORT_SYMBOL int control_close_session(control_session_handle_t, int);

    EXPORT_SYMBOL int control_is_stage_enabled(control_session_handle_t, const char *);
    EXPORT_SYMBOL void control_get_stage_flags(control_session_handle_t, const char *, control_stage_flags_t *);

    EXPORT_SYMBOL void control_get_scope(control_session_handle_t, control_scope_t *);
#else
    typedef u_int32_t (*PF_control_get_api_version)();

    typedef control_session_handle_t (*PF_control_open_session)(u_int64_t);
    typedef int (*PF_control_close_session)(control_session_handle_t, int);

    typedef int (*PF_control_is_stage_enabled)(control_session_handle_t, const char *);
    typedef int (*PF_control_get_stage_flags)(control_session_handle_t, const char *, control_stage_flags_t *);

    typedef void (*PF_control_get_scope)(control_session_handle_t, control_scope_t *);
#endif

#ifdef __cplusplus
}
#endif

#endif /* IBDIAGNET_CONTROL_API_H */
