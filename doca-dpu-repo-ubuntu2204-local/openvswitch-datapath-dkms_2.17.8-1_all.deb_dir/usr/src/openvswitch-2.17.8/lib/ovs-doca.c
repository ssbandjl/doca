/*
 * Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES, ALL RIGHTS RESERVED.
 *
 * This software product is a proprietary product of NVIDIA CORPORATION &
 * AFFILIATES (the "Company") and all right, title, and interest in and to the
 * software product, including all associated intellectual property rights, are
 * and shall remain exclusively with the Company.
 *
 * This software product is governed by the End User License Agreement
 * provided with the software product.
 *
 */

#include <config.h>

#include <rte_mtr.h>

#include <doca_flow.h>
#include <doca_log.h>
#include <doca_version.h>

#include "conntrack-offload.h"
#include "dpdk.h"
#include "netdev-offload.h"
#include "netdev-offload-provider.h"
#include "openvswitch/vlog.h"
#include "ovs-doca.h"
#include "unixctl.h"

VLOG_DEFINE_THIS_MODULE(ovs_doca);

#define OVS_DOCA_CONGESTION_THRESHOLD_DEFAULT 80
#define OVS_DOCA_CONGESTION_THRESHOLD_MIN 30
#define OVS_DOCA_CONGESTION_THRESHOLD_MAX 90

/* Indicates successful initialization of DOCA. */
static atomic_bool doca_enabled = ATOMIC_VAR_INIT(false);
static atomic_bool doca_initialized = ATOMIC_VAR_INIT(false);
static FILE *log_stream = NULL;       /* Stream for DOCA log redirection */
static struct doca_log_backend *ovs_doca_log = NULL;
unsigned int doca_congestion_threshold = OVS_DOCA_CONGESTION_THRESHOLD_DEFAULT;

bool ovs_doca_async = true;
/* Size of control pipes. If zero, DOCA uses its default value. */
uint32_t ctl_pipe_size = 0;
uint32_t ctl_pipe_infra_size = 0;

atomic_bool ovs_doca_eswitch_active_ids[OVS_DOCA_MAX_ESW];

/* Memorization counters are stored in pairs (red, green) in a contiguous array:
 *   ESW_0              ESW_1              ...
 *  [<------ N ------>][<------ N ------>][...]
 * Where N = OVS_DOCA_MAX_METER_COUNTERS_PER_ESW.
 */
static struct {
    struct doca_flow_query red;
    struct doca_flow_query green;
} mtr_last_stats[OVS_DOCA_MAX_METERS];

static inline unsigned int
ovs_doca_meter_counters_per_esw_base_id(void)
{
    return ovs_doca_max_ct_counters_per_esw();
}

#define MAX_PORT_STR_LEN 128

bool
ovs_doca_enabled(void)
{
    bool enabled;

    atomic_read_relaxed(&doca_enabled, &enabled);
    return enabled;
}

bool
ovs_doca_initialized(void)
{
    bool initialized;

    atomic_read_relaxed(&doca_initialized, &initialized);
    return initialized;
}

static ssize_t
ovs_doca_log_write(void *c OVS_UNUSED, const char *buf, size_t size)
{
    static struct vlog_rate_limit rl = VLOG_RATE_LIMIT_INIT(600, 600);
    static struct vlog_rate_limit dbg_rl = VLOG_RATE_LIMIT_INIT(600, 600);

    switch (doca_log_level_get_global_sdk_limit()) {
        case DOCA_LOG_LEVEL_TRACE:
            VLOG_DBG_RL(&dbg_rl, "%.*s", (int) size, buf);
            break;
        case DOCA_LOG_LEVEL_DEBUG:
            VLOG_DBG_RL(&dbg_rl, "%.*s", (int) size, buf);
            break;
        case DOCA_LOG_LEVEL_INFO:
            VLOG_INFO_RL(&rl, "%.*s", (int) size, buf);
            break;
        case DOCA_LOG_LEVEL_WARNING:
            VLOG_WARN_RL(&rl, "%.*s", (int) size, buf);
            break;
        case DOCA_LOG_LEVEL_ERROR:
            VLOG_ERR_RL(&rl, "%.*s", (int) size, buf);
            break;
        case DOCA_LOG_LEVEL_CRIT:
            VLOG_EMER("%.*s", (int) size, buf);
            break;
        default:
            OVS_NOT_REACHED();
    }

    return size;
}

static cookie_io_functions_t ovs_doca_log_func = {
    .write = ovs_doca_log_write,
};

static void
ovs_doca_unixctl_mem_stream(struct unixctl_conn *conn, int argc OVS_UNUSED,
                            const char *argv[] OVS_UNUSED, void *aux)
{
    void (*callback)(FILE *) = aux;
    char *response = NULL;
    FILE *stream;
    size_t size;

    stream = open_memstream(&response, &size);
    if (!stream) {
        response = xasprintf("Unable to open memstream: %s.",
                             ovs_strerror(errno));
        unixctl_command_reply_error(conn, response);
        goto out;
    }

    callback(stream);
    fclose(stream);
    unixctl_command_reply(conn, response);
out:
    free(response);
}

static const char * const levels[] = {
    [DOCA_LOG_LEVEL_CRIT]    = "critical",
    [DOCA_LOG_LEVEL_ERROR]   = "error",
    [DOCA_LOG_LEVEL_WARNING] = "warning",
    [DOCA_LOG_LEVEL_INFO]    = "info",
    [DOCA_LOG_LEVEL_DEBUG]   = "debug",
    [DOCA_LOG_LEVEL_TRACE]   = "trace",
};

static int
ovs_doca_parse_log_level(const char *s)
{
    int i;

    for (i = 0; i < ARRAY_SIZE(levels); ++i) {
        if (levels[i] && !strcmp(s, levels[i])) {
            return i;
        }
    }
    return -1;
}

static const char *
ovs_doca_log_level_to_str(uint32_t log_level)
{
    int i;

    for (i = 0; i < ARRAY_SIZE(levels); ++i) {
        if (i == log_level && levels[i]) {
            return levels[i];
        }
    }
    return NULL;
}

static void
ovs_doca_unixctl_log_set(struct unixctl_conn *conn, int argc,
                         const char *argv[], void *aux OVS_UNUSED)
{
    int level = DOCA_LOG_LEVEL_DEBUG;

    /* With no argument, level is set to 'debug'. */

    if (argc == 2) {
        const char *level_string;

        level_string = argv[1];
        level = ovs_doca_parse_log_level(level_string);
        if (level == -1) {
            char *err_msg;

            err_msg = xasprintf("invalid log level: '%s'", level_string);
            unixctl_command_reply_error(conn, err_msg);
            free(err_msg);
            return;
        }
    }

    doca_log_level_set_global_sdk_limit(level);
    unixctl_command_reply(conn, NULL);
}

static void
ovs_doca_log_dump(FILE *stream)
{
    uint32_t log_level;

    log_level = doca_log_level_get_global_sdk_limit();
    fprintf(stream, "DOCA log level is %s", ovs_doca_log_level_to_str(log_level));
}

static void
ovs_doca_dynamic_config(const struct smap *config)
{
    static const char *mode_names[] = {
        [0] = "synchronous",
        [1] = "asynchronous",
    };
    uint32_t req_ctl_pipe_size;
    bool req_doca_async;

    if (!smap_get_bool(config, "doca-init", false)) {
        return;
    }

    /* Once the user request a DOCA run, we must modify all future logic
     * (DPDK port probing) to take it into account, even if it results
     * in a failure. Once set, this value won't change. */
    if (!ovs_doca_enabled()) {
        atomic_store(&doca_enabled, true);
    }

    req_doca_async = smap_get_bool(config, "doca-async", true);
    if (req_doca_async != ovs_doca_async) {
        VLOG_INFO("Changing DOCA insertion mode from %s to %s.",
                  mode_names[!!ovs_doca_async], mode_names[!!req_doca_async]);

        ovs_doca_async = req_doca_async;
    } else {
        VLOG_INFO_ONCE("DOCA insertion mode is %s",
                       mode_names[!!ovs_doca_async]);
    }

    req_ctl_pipe_size = smap_get_uint(config, "ctl-pipe-size", 0);
    if (req_ctl_pipe_size != ctl_pipe_size) {
        VLOG_INFO("Changing DOCA ctl-pipe size from %"PRIu32" to %"PRIu32,
                  ctl_pipe_size, req_ctl_pipe_size);

        ctl_pipe_size = req_ctl_pipe_size;
    } else {
        VLOG_INFO_ONCE("DOCA ctl-pipe-size is %"PRIu32, ctl_pipe_size);
    }
}

int
ovs_doca_init(const struct smap *ovs_other_config)
{
    static struct ovsthread_once once_enable = OVSTHREAD_ONCE_INITIALIZER;
    unsigned int nb_threads = DEFAULT_OFFLOAD_THREAD_NB;
    unsigned int req_doca_congestion_threshold;
    uint32_t req_ctl_pipe_infra_size;
    struct doca_flow_cfg cfg = {};
    static bool enabled = false;
    uint16_t rss_queues;
    doca_error_t err;

    /* Dynamic configuration:
     * This section can be modified without restarting the process. */

    ovs_doca_dynamic_config(ovs_other_config);

    /* Static configuration:
     * This section is set once, restart is required after a change. */

    if (!ovsthread_once_start(&once_enable)) {
        return 0;
    }

    req_ctl_pipe_infra_size = smap_get_uint(ovs_other_config,
                                            "ctl-pipe-infra-size", UINT32_MAX);
    if (req_ctl_pipe_infra_size != UINT32_MAX) {
        VLOG_INFO("Changing DOCA ctl-pipe infra size from %"PRIu32" to %"PRIu32,
                  ctl_pipe_infra_size, req_ctl_pipe_infra_size);

        ctl_pipe_infra_size = req_ctl_pipe_infra_size;
    } else {
        ctl_pipe_infra_size = ctl_pipe_size;
    }
    VLOG_INFO("DOCA ctl-pipe-infra_size is %"PRIu32, ctl_pipe_infra_size);

    log_stream = fopencookie(NULL, "w+", ovs_doca_log_func);
    if (log_stream == NULL) {
        VLOG_ERR("Can't redirect DOCA log: %s.", ovs_strerror(errno));
    } else {
        /* Create a logger backend that prints to the redirected log */
        err = doca_log_backend_create_with_file_sdk(log_stream, &ovs_doca_log);
        if (err != DOCA_SUCCESS) {
            ovsthread_once_done(&once_enable);
            return EXIT_FAILURE;
        }
        doca_log_level_set_global_sdk_limit(DOCA_LOG_LEVEL_WARNING);
    }
    unixctl_command_register("doca/log-set", "{level}. level=critical/error/"
                             "warning/info/debug", 0, 1,
                             ovs_doca_unixctl_log_set, NULL);
    unixctl_command_register("doca/log-get", "", 0, 0,
                             ovs_doca_unixctl_mem_stream, ovs_doca_log_dump);

    if (!enabled && ovs_other_config &&
        smap_get_bool(ovs_other_config, "doca-init", false)) {
        /* Set dpdk-init to be true if not already set */
        smap_replace(CONST_CAST(struct smap *, ovs_other_config),
                     "dpdk-init", "true");
        dpdk_init(ovs_other_config);

        /* OVS-DOCA configuration happens earlier than dpif-netdev's.
         * To avoid reorganizing them, read the relevant item directly. */
        conntrack_offload_config(ovs_other_config);

        /* Due to limitation in doca, only one offload thread is currently
         * supported.
         * */
        if (smap_get_uint(ovs_other_config,"n-offload-threads", 1) !=
            nb_threads) {
            smap_replace(CONST_CAST(struct smap *, ovs_other_config),
                         "n-offload-threads", "1");
            VLOG_WARN_ONCE("Only %u offload thread is currently supported with"
                           "doca, ignoring n-offload-threads configuration",
                           nb_threads);
        }

        req_doca_congestion_threshold =
            smap_get_uint(ovs_other_config, "doca-congestion-threshold",
                          OVS_DOCA_CONGESTION_THRESHOLD_DEFAULT);
        if (req_doca_congestion_threshold != doca_congestion_threshold) {
            if (req_doca_congestion_threshold < OVS_DOCA_CONGESTION_THRESHOLD_MIN ||
                req_doca_congestion_threshold > OVS_DOCA_CONGESTION_THRESHOLD_MAX ) {
                VLOG_WARN("doca-congestion-threshold (%u) is not within expected"
                           " range (%u, %u)", req_doca_congestion_threshold,
                           OVS_DOCA_CONGESTION_THRESHOLD_MIN,
                           OVS_DOCA_CONGESTION_THRESHOLD_MAX);
            }
            VLOG_INFO("Changing DOCA doca-congestion-threshold from %u to %u",
                      doca_congestion_threshold, req_doca_congestion_threshold);

            doca_congestion_threshold = req_doca_congestion_threshold;
        } else {
            VLOG_INFO_ONCE("DOCA doca-congestion-threhold is %u",
                           doca_congestion_threshold);
        }

        cfg.pipe_queues = smap_get_uint(ovs_other_config, "n-offload-threads", 1);
        cfg.resource.nb_counters = OVS_DOCA_MAX_MEGAFLOWS_COUNTERS;
        cfg.mode_args = "switch,hws,cpds";
        cfg.queue_depth = OVS_DOCA_QUEUE_DEPTH;
        cfg.cb = ovs_doca_entry_process_cb;
        cfg.pipe_process_cb = ovs_doca_pipe_process_cb;
        cfg.rss.nr_queues = 1;
        rss_queues = 0;
        cfg.rss.queues_array = &rss_queues;
        /* Set the sum of counters we want for both ports */
        cfg.nr_shared_resources[DOCA_FLOW_SHARED_RESOURCE_COUNT] =
            ovs_doca_max_shared_counters();
        cfg.nr_shared_resources[DOCA_FLOW_SHARED_RESOURCE_METER] =
            OVS_DOCA_MAX_METERS;

        VLOG_INFO("DOCA Enabled - initializing...");
        err = doca_flow_init(&cfg);
        if (err) {
            VLOG_ERR("Error initializing doca flow offload. Error %d (%s)\n",
                    err, doca_error_get_descr(err));

            ovsthread_once_done(&once_enable);
            ovs_abort(err, "Cannot init DOCA");
            return err;
        }

        enabled = true;
        VLOG_INFO("DOCA Enabled - initialized");
    }

    ovsthread_once_done(&once_enable);
    atomic_store_relaxed(&doca_initialized, enabled);

    return 0;
}

void *
ovs_doca_port_create(uint16_t port_id)
{
    struct doca_flow_port_cfg port_cfg;
    char port_id_str[MAX_PORT_STR_LEN];
    struct doca_flow_port *port;
    doca_error_t err;

    memset(&port_cfg, 0, sizeof(port_cfg));

    port_cfg.port_id = port_id;
    port_cfg.type = DOCA_FLOW_PORT_DPDK_BY_ID;
    snprintf(port_id_str, MAX_PORT_STR_LEN, "%d", port_cfg.port_id);
    port_cfg.devargs = port_id_str;

    err = doca_flow_port_start(&port_cfg, &port);
    if (err) {
        VLOG_ERR("Failed to start doca flow port_id %"PRIu16". Error: %d (%s)",
                 port_id, err, doca_error_get_descr(err));

        return NULL;
    }

    return (void *)port;
}

int
ovs_doca_port_destroy(void *port)
{
    struct doca_flow_port *doca_port = port;

    return doca_flow_port_stop(doca_port);
}

void
ovs_doca_status(const struct ovsrec_open_vswitch *cfg)
{
    if (!cfg) {
        return;
    }
    ovsrec_open_vswitch_set_doca_initialized(cfg, ovs_doca_initialized());
    ovsrec_open_vswitch_set_doca_version(cfg, doca_version_runtime());
}

void
print_doca_version(void)
{
    printf("DOCA %s\n", doca_version_runtime());
}

const char *
ovs_doca_get_version(void)
{
    return doca_version_runtime();
}

static void
fill_meter_profile(struct doca_flow_shared_resource_cfg *doca_cfg,
                   struct ofputil_meter_config *config)
{
    doca_cfg->domain = DOCA_FLOW_PIPE_DOMAIN_DEFAULT;

    if (config->flags & OFPMF13_PKTPS) {
        doca_cfg->meter_cfg.limit_type = DOCA_FLOW_METER_LIMIT_TYPE_PACKETS;
        doca_cfg->meter_cfg.cir = config->bands[0].rate;
        doca_cfg->meter_cfg.cbs = config->bands[0].burst_size;
    } else {
        doca_cfg->meter_cfg.limit_type = DOCA_FLOW_METER_LIMIT_TYPE_BYTES;
        /* Convert from kilobits per second to bytes per second */
        doca_cfg->meter_cfg.cir = ((uint64_t) config->bands[0].rate) * 125;
        doca_cfg->meter_cfg.cbs = ((uint64_t) config->bands[0].burst_size) * 125;
    }
}

uint32_t
ovs_doca_meter_id(uint32_t dp_meter_id, uint32_t esw_id)
{
    return esw_id * OVS_DOCA_MAX_METERS_PER_ESW + dp_meter_id;
}

int
ovs_doca_create_meter(uint32_t of_meter_id,
                      struct ofputil_meter_config *config,
                      struct rte_mtr_error *error)
{
    struct doca_flow_shared_resource_cfg meter_cfg;
    uint32_t doca_mtr_id;
    uint32_t esw_id;
    int ret;

    if (config->n_bands != 1) {
        return -1;
    }

    memset(&meter_cfg, 0, sizeof meter_cfg);
    fill_meter_profile(&meter_cfg, config);

    /* At this stage it's not known with which eswitch the meter will be used,
     * thus need to configure meter for all available eswitches.
     */
    for (esw_id = 0; esw_id < OVS_DOCA_MAX_ESW; esw_id++) {
        /* id is determine by both the upper layer id, and the esw_id. */
        doca_mtr_id = ovs_doca_meter_id(of_meter_id, esw_id);

        ret = doca_flow_shared_resource_cfg(DOCA_FLOW_SHARED_RESOURCE_METER,
                                            doca_mtr_id, &meter_cfg);
        if (ret != DOCA_SUCCESS) {
            VLOG_ERR("Failed to configure shared meter id %d for eswitch%d, err %d - %s",
                     of_meter_id, esw_id, ret, doca_error_get_descr(ret));
            if (error) {
                error->type = RTE_MTR_ERROR_TYPE_UNSPECIFIED;
                error->message = doca_error_get_descr(ret);
                break;
            }
        }
    }

    return ret;
}

static int
color_offset(enum doca_flow_meter_color meter_color)
{
    if (meter_color == DOCA_FLOW_METER_COLOR_RED) {
        return 0;
    }
    if (meter_color == DOCA_FLOW_METER_COLOR_GREEN) {
        return 1;
    }
    /* Yellow color is not used at the moment */
    OVS_NOT_REACHED();
    return -1;
}

uint32_t
ovs_doca_get_post_meter_counter_id(uint32_t doca_meter_id,
                                   enum doca_flow_meter_color meter_color)
{
    uint32_t eswitch_id = doca_meter_id / OVS_DOCA_MAX_METERS_PER_ESW;
    int offset = color_offset(meter_color);

    return eswitch_id * ovs_doca_max_shared_counters_per_esw() +
           ovs_doca_meter_counters_per_esw_base_id() +
           (doca_meter_id - (eswitch_id * OVS_DOCA_MAX_METERS_PER_ESW)) *
           OVS_DOCA_MAX_METER_COLORS - offset;
}

static int
query_counter(uint32_t counter_id, struct doca_flow_query *counter_stats,
              struct rte_mtr_error *error)
{
    struct doca_flow_shared_resource_result query_results;
    doca_error_t ret;

    memset(&query_results, 0, sizeof query_results);

    ret = doca_flow_shared_resources_query(DOCA_FLOW_SHARED_RESOURCE_COUNT,
                                           &counter_id, &query_results, 1);
    if (ret != DOCA_SUCCESS) {
        VLOG_ERR("Failed to query shared meter counter id %u: %s",
                 counter_id, doca_error_get_descr(ret));
        error->type = RTE_MTR_ERROR_TYPE_STATS;
        error->message = doca_error_get_descr(ret);
        return -1;
    }

    memcpy(counter_stats, &query_results.counter, sizeof *counter_stats);

    return 0;
}

int
ovs_doca_delete_meter(uint16_t port_id OVS_UNUSED, uint32_t dp_meter_id,
                      struct rte_mtr_error *error)
{
    struct doca_flow_query red_stats, green_stats;
    uint32_t red_counter_id, green_counter_id;
    uint32_t doca_mtr_id;
    uint32_t esw_id;
    int ret;

    /* DOCA shared meter cannot be unbound explicitly, instead DOCA handles it
    * internally, when the port or the pipe, where the meter has been bound to,
    * is destroyed.
    * Same holds true for meter counters. This means that the next time user
    * creates a meter with the same ID that has been used previously, the
    * statistics will be reported incorrectly as the meter counters cannot be
    * reset while they're bound.
    * As a workaround counter values are saved when meter is deleted and used
    * to offset statistics for the next meter that re-uses the same ID.
    */
    OVS_DOCA_FOREACH_ACTIVE_ESWITCH (esw_id) {
        /* id is determine by both the upper layer id, and the esw_id. */
        doca_mtr_id = ovs_doca_meter_id(dp_meter_id, esw_id);

        red_counter_id = ovs_doca_get_post_meter_counter_id(doca_mtr_id,
                                                            DOCA_FLOW_METER_COLOR_RED);
        ret = query_counter(red_counter_id, &red_stats, error);
        if (ret) {
            VLOG_ERR("Failed to query red meter counter, meter id %u for eswitch%d",
                     doca_mtr_id, esw_id);
            return -1;
        }
        mtr_last_stats[doca_mtr_id].red = red_stats;

        green_counter_id = ovs_doca_get_post_meter_counter_id(doca_mtr_id,
                                                              DOCA_FLOW_METER_COLOR_GREEN);
        ret = query_counter(green_counter_id, &green_stats, error);
        if (ret) {
            VLOG_ERR("Failed to query green meter counter, meter id %u for eswitch%d",
                     doca_mtr_id, esw_id);
            return -1;
        }
        mtr_last_stats[doca_mtr_id].green = green_stats;
    }

    return 0;
}

int
ovs_doca_mtr_stats_read(uint16_t port_id OVS_UNUSED, uint32_t dp_meter_id,
                        struct rte_mtr_stats *stats,
                        struct rte_mtr_error *error)
{
    struct doca_flow_query *last_red_stats, *last_green_stats;
    struct doca_flow_query red_stats, green_stats;
    uint32_t red_counter_id, green_counter_id;
    uint64_t red_bytes, green_bytes;
    uint64_t red_pkts, green_pkts;
    uint32_t doca_mtr_id;
    uint32_t esw_id = 0; /* Only first eswitch is supported for now */
    int ret;

    /* id is determine by both the upper layer id, and the esw_id. */
    doca_mtr_id = ovs_doca_meter_id(dp_meter_id, esw_id);

    last_red_stats = &mtr_last_stats[doca_mtr_id].red;
    last_green_stats = &mtr_last_stats[doca_mtr_id].green;

    red_counter_id =
        ovs_doca_get_post_meter_counter_id(doca_mtr_id,
                                           DOCA_FLOW_METER_COLOR_RED);
    green_counter_id =
        ovs_doca_get_post_meter_counter_id(doca_mtr_id,
                                           DOCA_FLOW_METER_COLOR_GREEN);

    ret = query_counter(red_counter_id, &red_stats, error);
    if (ret) {
        VLOG_ERR("Failed to query red counter for meter id %u", doca_mtr_id);
        return -1;
    }

    ret = query_counter(green_counter_id, &green_stats, error);
    if (ret) {
        VLOG_ERR("Failed to query green counter for meter id %u", doca_mtr_id);
        return -1;
    }

    /* check for counter wraparound */
    red_pkts = ovs_u64_wrapsub(red_stats.total_pkts,
                               last_red_stats->total_pkts);
    red_bytes = ovs_u64_wrapsub(red_stats.total_bytes,
                                last_red_stats->total_bytes);
    green_pkts = ovs_u64_wrapsub(green_stats.total_pkts,
                                 last_green_stats->total_pkts);
    green_bytes = ovs_u64_wrapsub(green_stats.total_bytes,
                                  last_green_stats->total_bytes);

    stats->n_pkts[RTE_COLOR_GREEN] = green_pkts;
    stats->n_pkts[RTE_COLOR_YELLOW] = 0;
    stats->n_pkts[RTE_COLOR_RED] = red_pkts;

    stats->n_bytes[RTE_COLOR_GREEN] = green_bytes;
    stats->n_bytes[RTE_COLOR_YELLOW] = 0;
    stats->n_bytes[RTE_COLOR_RED] = red_bytes;

    stats->n_pkts_dropped = red_pkts;
    stats->n_bytes_dropped = red_bytes;

    return 0;
}

unsigned int
ovs_doca_max_ct_conns(void)
{
    return conntrack_offload_size();
}
