/* SPDX-License-Identifier: BSD-3-Clause
 * Copyright 2020 Mellanox Technologies, Ltd
 */

#ifndef _RTE_SFT_H_
#define _RTE_SFT_H_

/**
 * @file
 *
 * RTE SFT API
 *
 * Defines RTE SFT APIs for Statefull Flow Table library.
 *
 * The SFT lib is part of the ethdev class, the reason for this is that the main
 * idea is to leverage the HW offload that the ethdev allow using the rte_flow.
 *
 * SFT General description:
 * SFT library provides a framework for applications that need to maintain
 * context across different packets of the connection.
 * Examples for such applications:
 * - Next-generation firewalls
 * - Intrusion detection/prevention systems (IDS/IPS): Suricata, Snort
 * - SW/Virtual Switching: OVS
 * The goals of the SFT library:
 * - Accelerate flow recognition & its context retrieval for further look-aside
 *   processing.
 * - Enable context-aware flow handling offload.
 *
 * The SFT is designed to use HW offload to get the best performance.
 * This is done on two levels. The first one is marking the packet with flow id
 * to speed the lookup of the flow in the data structure.
 * The second is done by connecting the SFT results to the rte_flow for
 * continuing packet process.
 *
 * Definitions and Abbreviations:
 * - 5-tuple: defined by:
 *     -- Source IP address
 *     -- Source port
 *     -- Destination IP address
 *     -- Destination port
 *     -- IP protocol number
 * - 7-tuple: 5-tuple, zone and port (see struct rte_sft_7tuple)
 * - 5/7-tuple: 5/7-tuple of the packet from connection initiator
 * - revers 5/7-tuple: 5/7-tuple of the packet from connection initiate
 * - application: SFT library API consumer
 * - APP: see application
 * - CID: client ID
 * - CT: connection tracking
 * - FID: Flow identifier
 * - FIF: First In Flow
 * - Flow: defined by 7-tuple and its reverse i.e. flow is bidirectional
 * - SFT: Stateful Flow Table
 * - user: see application
 * - zone: additional user defined value used as differentiator for
 *         connections having same 5-tuple (for example different VXLAN
 *         connections with same inner 5-tuple).
 *
 * SFT components:
 *
 * +-----------------------------------+
 * | RTE flow                          |
 * |                                   |
 * | +-------------------------------+ |  +----------------+
 * | | group X                       | |  | RTE_SFT        |
 * | |                               | |  |                |
 * | | +---------------------------+ | |  |                |
 * | | | rule ...                  | | |  |                |
 * | | | .                         | | |  +-----------+----+
 * | | | .                         | | |              |
 * | | | .                         | | |          entry
 * | | +---------------------------+ | |            create
 * | | | rule                      | | |              |
 * | | |   patterns ...            +---------+        |
 * | | |   actions                 | | |     |        |
 * | | |     SFT (zone=Z)          | | |     |        |
 * | | |     JUMP (group=Y)        | | |  lookup      |
 * | | +---------------------------+ | |    zone=Z,   |
 * | | | rule ...                  | | |    5tuple    |
 * | | | .                         | | |     |        |
 * | | | .                         | | |  +--v-------------+
 * | | | .                         | | |  | SFT       |    |
 * | | |                           | | |  |           |    |
 * | | +---------------------------+ | |  |        +--v--+ |
 * | |                               | |  |        |     | |
 * | +-------------------------------+ |  |        | PMD | |
 * |                                   |  |        |     | |
 * |                                   |  |        +-----+ |
 * | +-------------------------------+ |  |                |
 * | | group Y                       | |  |                |
 * | |                               | |  | set state      |
 * | | +---------------------------+ | |  | set data       |
 * | | | rule                      | | |  +--------+-------+
 * | | |   patterns                | | |           |
 * | | |     SFT (state=UNDEFINED) | | |           |
 * | | |   actions RSS             | | |           |
 * | | +---------------------------+ | |           |
 * | | | rule                      | | |           |
 * | | |   patterns                | | |           |
 * | | |     SFT (state=INVALID)   | <-------------+
 * | | |   actions DROP            | | |  forward
 * | | +---------------------------+ | |    group=Y
 * | | | rule                      | | |
 * | | |   patterns                | | |
 * | | |     SFT (state=ACCEPTED)  | | |
 * | | |   actions PORT            | | |
 * | | +---------------------------+ | |
 * | |  ...                          | |
 * | |                               | |
 * | +-------------------------------+ |
 * |  ...                              |
 * |                                   |
 * +-----------------------------------+
 *
 * SFT as datastructure:
 * SFT can be treated as datastructure maintaining flow context across its
 * lifetime. SFT flow entry represents bidirectional network flow and defined by
 * 7-tuple & its reverse 7-tuple.
 * Each entry in SFT has:
 * - FID: 1:1 mapped & used as entry handle & encapsulating internal
 *   implementation of the entry.
 * - State: user-defined value attached to each entry, the only library
 *   reserved value for state unset (the actual value defined by SFT
 *   configuration). The application should define flow state encodings and
 *   set it for flow via rte_sft_flow_set_ctx() than what actions should be
 *   applied on packets can be defined via related RTE flow rule matching SFT
 *   state (see rules in SFT components diagram above).
 * - Timestamp: for the last seen in flow packet used for flow aging mechanism
 *   implementation.
 * - Client Objects: user-defined flow contexts attached as opaques to flow.
 * - Acceleration & offloading - utilize RTE flow capabilities, when supported
 *   (see action ``SFT``), for flow lookup acceleration and further
 *   context-aware flow handling offload.
 * - CT state: optionally for TCP connections CT state can be maintained
 *   (see enum rte_sft_flow_ct_state).
 * - Out of order TCP packets: optionally SFT can keep out of order TCP
 *   packets aside the flow context till the arrival of the missing in-order
 *   packet.
 *
 * RTE flow changes:
 * The SFT flow state (or context) for RTE flow is defined by fields of
 * struct rte_flow_item_sft.
 * To utilize SFT capabilities new item and action types introduced:
 * - item SFT: matching on SFT flow state (see RTE_FLOW_ITEM_TYPE_SFT).
 * - action SFT: retrieve SFT flow context and attached it to the processed
 *   packet (see RTE_FLOW_ACTION_TYPE_SFT).
 *
 * The contents of per port SFT serving RTE flow action ``SFT`` managed via
 * SFT PMD APIs (see struct rte_sft_ops).
 * The SFT flow state/context retrieval performed by user-defined zone ``SFT``
 * action argument and processed packet 5-tuple.
 * If in scope of action ``SFT`` there is no context/state for the flow in SFT
 * undefined state attached to the packet meaning that the flow is not
 * recognized by SFT, most probably FIF packet.
 *
 * Once the SFT state set for a packet it can match on item SFT
 * (see RTE_FLOW_ITEM_TYPE_SFT) and forwarding design can be done for the
 * packet, for example:
 * - if state value == x than queue for further processing by the application
 * - if state value == y than forward it to eth port (full offload)
 * - if state value == 'undefined' than queue for further processing by
 *   the application (handle FIF packets)
 *
 * Processing packets with SFT library:
 *
 * FIF packet:
 * To recognize upcoming packets of the SFT flow every FIF packet should be
 * forwarded to the application utilizing the SFT library. Non-FIF packets can
 * be processed by the application or its processing can be fully offloaded.
 * Processing of the packets in SFT library starts with rte_sft_process_mbuf
 * or rte_sft_process_mbuf_with_zone. If mbuf recognized as FIF application
 * should make a design to destroy flow or complete flow creation process in
 * SFT using rte_sft_flow_activate.
 *
 * Recognized SFT flow:
 * Once struct rte_sft_flow_status with valid fid field possessed by application
 * it can:
 * - mange client objects on it (see client_obj field in
 *   struct rte_sft_flow_status) using rte_sft_flow_<OP>_client_obj APIs
 * - analyze user-defined flow state and CT state.
 * - set flow state to be attached to the upcoming packets by action ``SFT``
 *   via struct rte_sft_flow_status API.
 * - decide to destroy flow via rte_sft_flow_destroy API.
 *
 * Flow aging:
 *
 * SFT library manages the aging for each flow. On flow creation, it's
 * assigned an aging value, the maximal number of seconds passed since the
 * last flow packet arrived, once exceeded flow considered aged.
 * The application notified of aged flow asynchronously via event queues.
 * The device and port IDs tuple to identify the event queue to enqueue
 * flow aged events passed on flow creation as arguments
 * (see rte_sft_flow_activate). It's the application responsibility to
 * initialize event queues and assign them to each flow for EOF event
 * notifications.
 * Aged EOF event handling:
 * - Should be considered as application responsibility.
 * - The last stage should be the release of the flow resources via
 *    rte_sft_flow_destroy API.
 * - All client objects should be removed from flow before the
 *   rte_sft_flow_destroy API call.
 * See the description of ret_sft_flow_destroy for an example of aged flow
 * handling.
 *
 * SFT API thread safety:
 *
 * Since the SFT lib is designed to work as part of the Fast-Path, The SFT
 * is not thread safe, in order to enable better working with multiple threads
 * the SFT lib uses the queue approach, where each queue can only be accessesd
 * by one thread while one thread can access multiple queues.
 *
 * SFT Library initialization and cleanup:
 *
 * SFT library should be considered as a single instance, preconfigured and
 * initialized via rte_sft_init() API.
 * SFT library resource deallocation and cleanup should be done via
 * rte_sft_init() API as a stage of the application termination procedure.
 */

#ifdef __cplusplus
extern "C" {
#endif

#include <rte_common.h>
#include <rte_config.h>
#include <rte_errno.h>
#include <rte_mbuf.h>
#include <rte_ethdev.h>
#include <rte_flow.h>

#define RTE_SFT_APP_ERR_STATE 0
#define RTE_SFT_APP_ERR_DATA_VAL 0

/**
 * L3/L4 5-tuple - src/dest IP and port and IP protocol.
 *
 * Used for flow/connection identification.
 */
__extension__
struct rte_sft_5tuple {
	union {
		struct {
			rte_be32_t src_addr; /**< IPv4 source address. */
			rte_be32_t dst_addr; /**< IPv4 destination address. */
		} ipv4;
		struct {
			uint8_t src_addr[16]; /**< IPv6 source address. */
			uint8_t dst_addr[16]; /**< IPv6 destination address. */
		} ipv6;
	};
	rte_be16_t src_port; /**< Source port. */
	rte_be16_t dst_port; /**< Destination port. */
	uint8_t proto; /**< IP protocol. */
	uint8_t is_ipv6: 1; /**< True for valid IPv6 fields. Otherwise IPv4. */
};

/**
 * Port flow identification.
 *
 * @p zone used for setups where 5-tuple is not enough to identify flow.
 * For example different VLANs/VXLANs may have similar 5-tuples.
 */
struct rte_sft_7tuple {
	struct rte_sft_5tuple flow_5tuple; /**< L3/L4 5-tuple. */
	uint32_t zone; /**< Zone assigned to flow. */
	uint16_t port_id; /** < Port identifier of Ethernet device. */
};

/**
 * Structure describes SFT library configuration
 */
struct rte_sft_conf {
	uint16_t nb_queues; /**< Preferred number of queues */
	uint32_t udp_aging; /**< UDP proto default aging in sec */
	uint32_t tcp_aging; /**< TCP proto default aging in sec */
	uint32_t default_aging; /**< All unlisted proto default aging in sec. */
	uint32_t nb_max_entries; /**< Max entries in SFT. */
	uint16_t nb_max_ipfrag; /**< Max IP fragments queue can store */
	uint16_t ipfrag_timeout; /**< timeout for IP defarag library */
	uint8_t app_data_len; /**< Number of uint32 of app data. */
	uint32_t support_partial_match: 1;
	/**< App can partial match on the data. */
	uint32_t reorder_enable: 1;
	/**< TCP packet reordering feature enabled bit. */
	uint32_t tcp_ct_enable: 1;
	/**< TCP connection tracking based on standard. */
	uint32_t ipfrag_enable: 1;
	uint32_t reserved: 28;
};

#define RTE_SFT_ACTION_INITIATOR_NAT (1ul << 0)
/**< NAT action should be done on the initiator traffic. */
#define RTE_SFT_ACTION_REVERSE_NAT (1ul << 1)
/**< NAT action should be done on the reverse traffic. */
#define RTE_SFT_ACTION_COUNT (1ul << 2) /**< Enable count action. */
#define RTE_SFT_ACTION_AGE (1ul << 3) /**< Enable ageing action. */

/**
 *  Structure that holds the action configuration.
 */
struct rte_sft_actions_specs {
	uint64_t actions; /**< Action flags. See RTE_SFT_ACTION_* */
	struct rte_sft_5tuple *initiator_nat;
	/**< The NAT configuration for the initiator flow. */
	struct rte_sft_5tuple *reverse_nat;
	/**< The NAT configuration for the reverse flow. */
	uint64_t aging; /**< the aging time out in sec. */
};

/**
 * Structure that holds the count data.
 */
struct rte_sft_query_data {
	uint64_t nb_bytes[2];
	/**< Number of bytes that passed in the flow,
	 * index 1 is initiator, index 0 is reply.
	 */
	uint64_t nb_packets[2];
	/**< Number of packets that passed in the flow,
	 * index 1 is initiator, index 0 is reply.
	 */
	uint32_t age; /**< Seconds passed since last seen packet. */
	uint32_t aging;
	/**< Flow considered aged once this age (seconds) reached. */
	uint32_t nb_bytes_valid: 1; /**< Number of bytes is valid. */
	uint32_t nb_packets_valid: 1; /* Number of packets is valid. */
	uint32_t nb_age_valid: 1; /* Age is valid. */
	uint32_t nb_aging_valid: 1; /* Aging is valid. */
	uint32_t reserved: 28;
};

/**
 * Connection tracking info
 */
enum sft_ct_info {
	SFT_CT_ERROR_UNSUPPORTED = INT8_MIN,
	SFT_CT_ERROR_BAD_PROTOCOL,
	SFT_CT_ERROR_TCP_SYN,
	SFT_CT_ERROR_TCP_FLAGS,
	SFT_CT_ERROR_TCP_SEND_SEQ,
	SFT_CT_ERROR_TCP_ACK_SEQ,
	SFT_CT_ERROR_TCP_RCV_WND_SIZE,
	SFT_CT_ERROR_SYS,
	SFT_CT_ERROR_NONE = 0,
	SFT_CT_RETRANSMIT,
	SFT_CT_RESET,
};

/**
 * Connection tracking states
 */
enum sft_ct_state {
	SFT_CT_STATE_NEW = 0,      /**< no FID */
	SFT_CT_STATE_ESTABLISHING, /**< connection establish in process */
	SFT_CT_STATE_TRACKING,     /**< full duplex data exchange */
	SFT_CT_STATE_HALF_DUPLEX,  /**< data flows in one direction only */
	SFT_CT_STATE_CLOSING,      /**< no data in any direction */
	SFT_CT_STATE_CLOSED,	   /**< confirmed termination from 2 peers */
	SFT_CT_STATE_OFFLOADED,    /**< offloaded connection */
	SFT_CT_STATE_ERROR	   /**< error connection state */
};

/**
 * Structure describes the state of the flow in SFT.
 */
struct rte_sft_flow_status {
	uint32_t fid; /**< SFT flow id. */
	uint32_t zone; /**< Zone for lookup in SFT */
	uint8_t state; /**< Application defined bidirectional flow state. */
	enum sft_ct_state proto_state; /**< The state based on the protocol. */
	enum sft_ct_info ct_info; /**< Connection tracking error */
	uint16_t proto; /**< L4 protocol. */
	/**< data_offset: mark valid data location in segment
	 * > 0 prefix shift (retransmit)
	 * < 0 suffix shift (rcv-window overrun)
	 */
	int16_t data_offset;
	/**< Connection tracking flow state, based on standard. */
	union {
		uint32_t nb_in_order_mbufs;
		/**< Number of in-order mbufs available for drain */
		uint32_t nb_ip_fragments;
		/**< Number of IP fragments ready for drain */
	};
	uint32_t proto_state_change: 1; /**< Protocol state was changed. */
	uint32_t protocol_error: 1; /**< packet does not fit protocol flow */
	uint32_t packet_error: 1; /**< Malformed packet */
	uint32_t fragmented: 1; /**< Last flow mbuf was fragmented. */
	uint32_t out_of_order: 1; /**< Last flow mbuf was out of order (TCP). */
	uint32_t activated: 1; /**< Flow was activated. */
	uint32_t zone_valid: 1; /**< Zone field is valid. */
	uint32_t offloaded: 1;
	/**< The connection is offload and no packet should be stored. */
	uint32_t initiator: 1; /**< marks if the mbuf is from the initiator. */
	uint32_t reserved: 23;
	uintptr_t ipfrag_ctx;
#ifdef SFT_CT_DEBUG
	uint32_t max_sent_seq;
#endif
	uint32_t data[];
	/**< Application data. The length is defined by the configuration. */
};

/**
 * Verbose error types.
 *
 * Most of them provide the type of the object referenced by struct
 * rte_flow_error.cause.
 */
enum rte_sft_error_type {
	RTE_SFT_ERROR_TYPE_NONE = 0, /**< No error. */
	RTE_SFT_ERROR_TYPE_UNSPECIFIED, /**< Cause unspecified. */
	RTE_SFT_ERROR_TYPE_FLOW_NOT_DEFINED, /**< The FID is not defined. */
	RTE_SFT_ERROR_TYPE_HASH_ERROR, /**< There was error in hash. */
	RTE_SFT_ERROR_CONN_TRACK,
	RTE_SFT_ERROR_IPFRAG,
	RTE_SFT_ERROR_CHECKSUM,
};

/**
 * Verbose error structure definition.
 *
 * This object is normally allocated by applications and set by SFT, the
 * message points to a constant string which does not need to be freed by
 * the application, however its pointer can be considered valid only as long
 * as its associated DPDK port remains configured. Closing the underlying
 * device or unloading the PMD invalidates it.
 *
 * Both cause and message may be NULL regardless of the error type.
 */
struct rte_sft_error {
	enum rte_sft_error_type type; /**< Cause field and error types. */
	const void *cause; /**< Object responsible for the error. */
	const char *message; /**< Human-readable error message. */
};

/**
 * @warning
 * @b EXPERIMENTAL: this API may change without prior notice.
 *
 * Get the list of SFT aged flows IDs
 *
 * @param queue
 *   The SFT queue number.
 * @param[in, out] fids
 *   The address of an array of pointers to the aged-out FIDs.
 * @param[in] nb_fids
 *   The length of FIDs array pointers.
 * @param[out] error
 *   Perform verbose error reporting if not NULL.
 *   Initialized in case of error only.
 *
 * @return
 *   if nb_fids is 0, return the number of all aged out SFT flows.
 *   if nb_fids is not 0 , return the number of aged out flows
 *   reported in the fids array, otherwise negative errno value.
 *
 * @see rte_flow_action_age
 * @see RTE_ETH_EVENT_SFT_AGED
 */
__rte_experimental
int
rte_sft_flow_get_aged_flows(uint16_t queue, uint32_t *fids,
			    uint32_t nb_fids, struct rte_sft_error *error);

/**
 * @warning
 * @b EXPERIMENTAL: this API may change without prior notice.
 *
 * Get SFT flow status, based on the fid.
 *
 * @param queue
 *   The sft queue number.
 * @param fid
 *   SFT flow ID.
 * @param[out] status
 *   Structure to dump actual SFT flow status.
 * @param[out] error
 *   Perform verbose error reporting if not NULL.
 *
 * @return
 *   0 on success, a negative errno value otherwise and rte_sft_error is set.
 */
__rte_experimental
int
rte_sft_flow_get_status(const uint16_t queue, const uint32_t fid,
			struct rte_sft_flow_status *status,
			struct rte_sft_error *error);

/**
 * @warning
 * @b EXPERIMENTAL: this API may change without prior notice.
 *
 * Set user defined data.
 *
 * @param queue
 *   The sft queue number.
 * @param fid
 *   SFT flow ID.
 * @param data
 *   User defined data. The len is defined at configuration time.
 * @param[out] error
 *   Perform verbose error reporting if not NULL.
 *
 * @return
 *   0 on success , a negative errno value otherwise and rte_sft_error is set.
 */
__rte_experimental
int
rte_sft_flow_set_data(uint16_t queue, uint32_t fid, const uint32_t *data,
		      struct rte_sft_error *error);

/**
 * @warning
 * @b EXPERIMENTAL: this API may change without prior notice.
 *
 * Set user defined state.
 *
 * @param queue
 *   The sft queue number.
 * @param fid
 *   SFT flow ID.
 * @param state
 *   User state.
 * @param[out] error
 *   Perform verbose error reporting if not NULL.
 *
 * @return
 *   0 on success , a negative errno value otherwise and rte_sft_error is set.
 */
__rte_experimental
int
rte_sft_flow_set_state(uint16_t queue, uint32_t fid, const uint8_t state,
		       struct rte_sft_error *error);

/**
 * @warning
 * @b EXPERIMENTAL: this API may change without prior notice.
 *
 * Mark the connection as offloaded. This will result in that the SFT
 * will disable the out of order, fragmentation and any other feature that
 * depends on the arrival of next packet.
 *
 * @param queue
 *   The sft queue number.
 * @param fid
 *   SFT flow ID.
 * @param offload
 *   set if flow is offloaded.
 * @param[out] error
 *   Perform verbose error reporting if not NULL.
 *
 * @return
 *   0 on success , a negative errno value otherwise and rte_sft_error is set.
 */
__rte_experimental
int
rte_sft_flow_set_offload(uint16_t queue, uint32_t fid, bool offload,
			 struct rte_sft_error *error);

/**
 * @warning
 * @b EXPERIMENTAL: this API may change without prior notice.
 *
 * Initialize SFT library instance.
 *
 * @param conf
 *   SFT library instance configuration.
 * @param[out] error
 *   Perform verbose error reporting if not NULL.
 *
 * @return
 *   0 on success , a negative errno value otherwise and rte_sft_error is set.
 */
__rte_experimental
int
rte_sft_init(const struct rte_sft_conf *conf, struct rte_sft_error *error);

/**
 * @warning
 * @b EXPERIMENTAL: this API may change without prior notice.
 *
 * Finalize SFT library instance.
 * Cleanup & release allocated resources.
 *
 * @param[out] error
 *   Perform verbose error reporting if not NULL.
 *
 * @return
 *   0 on success , a negative errno value otherwise and rte_sft_error is set.
 */
__rte_experimental
int
rte_sft_fini(struct rte_sft_error *error);

/**
 * @warning
 * @b EXPERIMENTAL: this API may change without prior notice.
 *
 * Process mbuf received on RX queue.
 *
 * This function checks the mbuf against the SFT database and return the
 * connection status that this mbuf belongs to.
 *
 * If status.activated = 1 and status.offloaded = 0 the input mbuf is
 * considered consumed and the application is not allowed to use it or free it,
 * instead the application should use the mbuf pointed by the mbuf_out.
 * In case the mbuf is out of order or fragmented the mbuf_out will be NULL.
 *
 * If status.activated = 0 or status.offloaded = 1, the input mbuf is not
 * consumed and the mbuf_out will always be NULL.
 *
 * This function doesn't create new entry in the SFT.
 *
 * @param queue
 *   The sft queue number.
 * @param[in] mbuf_in
 *   mbuf to process; mbuf pointer considered 'consumed' and should not be used
 *   if status.activated and status.offload = 0.
 * @param[out] mbuf_out
 *   last processed not fragmented and in order mbuf.
 * @param[out] status
 *   Connection status based on the last in mbuf.
 *   SFT updates relevant status fields only.
 * @param[out] error
 *   Perform verbose error reporting if not NULL. Initialize in case of
 *   error only.
 *
 * @return
 *   0 on success , a negative errno value otherwise and rte_sft_error is set.
 */
__rte_experimental
int
rte_sft_process_mbuf(uint16_t queue, struct rte_mbuf *mbuf_in,
		     struct rte_mbuf **mbuf_out,
		     struct rte_sft_flow_status *status,
		     struct rte_sft_error *error);

/**
 * @warning
 * @b EXPERIMENTAL: this API may change without prior notice.
 *
 * Process mbuf received on RX queue while zone value provided by caller.
 *
 * The behaviour of this function is similar to rte_sft_process_mbuf except
 * the lookup in SFT procedure. The lookup in SFT always done by the *zone*
 * arg and 5-tuple, extracted form mbuf outer header contents.
 *
 * @see rte_sft_process_mbuf
 *
 * @param queue
 *   The sft queue number.
 * @param[in] mbuf_in
 *   mbuf to process; mbuf pointer considered 'consumed' and should not be used
 *   if status.activated and status.offload = 0.
 * @param zone
 *   The requested zone.
 * @param[out] mbuf_out
 *   last processed not fragmented and in order mbuf.
 * @param[out] status
 *   Connection status based on the last in mbuf.
 * @param[out] error
 *   Perform verbose error reporting if not NULL. Initialize in case of
 *   error only.
 *
 * @return
 *   0 on success , a negative errno value otherwise and rte_sft_error is set.
 */
__rte_experimental
int
rte_sft_process_mbuf_with_zone(uint16_t queue, struct rte_mbuf *mbuf_in,
			       uint32_t zone, struct rte_mbuf **mbuf_out,
			       struct rte_sft_flow_status *status,
			       struct rte_sft_error *error);

/**
 * @warning
 * @b EXPERIMENTAL: this API may change without prior notice.
 *
 * Drain next in order mbuf.
 *
 * This function behaves similar to rte_sft_process_mbuf() but acts on packets
 * accumulated in SFT flow due to missing in order packet. Processing done on
 * single mbuf at a time and `in order`. Other than above the behavior is
 * same as of rte_sft_process_mbuf for flow defined & activated & mbuf isn't
 * fragmented & 'in order'. This function should be called when
 * rte_sft_process_mbuf or rte_sft_process_mbuf_with_zone sets
 * status->nb_in_order_mbufs output param !=0 and until
 * status->nb_in_order_mbufs == 0.
 * Flow should be locked by caller (see rte_sft_flow_lock).
 *
 * @param queue
 *   The sft queue number.
 * @param fid
 *   SFT flow ID.
 * @param[out] mbuf_out
 *   last processed not fragmented and in order mbuf.
 * @param nb_out
 *   Number of buffers to be drained.
 * @param initiator
 *   true packets that will be drained belongs to the initiator.
 * @param[out] status
 *   Connection status based on the last mbuf that was drained.
 * @param[out] error
 *   Perform verbose error reporting if not NULL. Initialize in case of
 *   error only.
 *
 * @return
 *   The number of mbufs that were drained, negative value in case
 *   of error and rte_sft_error is set.
 */
__rte_experimental
int
rte_sft_drain_mbuf(uint16_t queue, uint32_t fid,
		   struct rte_mbuf **mbuf_out, uint16_t nb_out,
		   bool initiator, struct rte_sft_flow_status *status,
		   struct rte_sft_error *error);

/**
 * @warning
 * @b EXPERIMENTAL: this API may change without prior notice.
 *
 * Activate flow in SFT.
 *
 * This function creates an entry in the SFT for this connection.
 * The reasons for 2 phase flow creation procedure:
 * 1. Missing reverse flow - flow context is shared for both flow directions
 *    i.e. in order maintain bidirectional flow context in RTE SFT packets
 *    arriving from both directions should be identified as packets of the
 *    RTE SFT flow. Consequently, before the creation of the SFT flow caller
 *    should provide reverse flow direction 7-tuple.
 * 2. The caller of rte_sft_process_mbuf/rte_sft_process_mbuf_with_zone should
 *   be notified that arrived mbuf is first in flow & decide whether to
 *   create a new flow or disregard this packet.
 * This function completes the creation of the bidirectional SFT flow & creates
 * entry for 7-tuple on SFT PMD defined by the tuple port for both
 * initiator/initiate 7-tuples.
 * Flow aging, connection tracking state & out of order handling will be
 * initialized according to the content of the *mbuf_in* passes to
 * rte_sft_process_mbuf/_with_zone during phase 1 of flow creation.
 * Once this function returns upcoming calls rte_sft_process_mbuf/_with_zone
 * with 7-tuple or its reverse will return the handle to this flow.
 * Flow should be locked by the caller (see rte_sft_flow_lock).
 *
 * @param queue
 *   The SFT queue.
 * @param zone
 *   The requested zone.
 * @param[in] mbuf_in
 *   mbuf to process; mbuf pointer considered 'consumed' and should not be used
 *   after successful call to this function.
 * @param reverse_tuple
 *   Expected response flow 7-tuple.
 * @param state
 *   User defined state to set.
 * @param data
 *   User defined data, the len is configured during sft init.
 * @param proto_enable
 *   Enables maintenance of status->proto_state connection tracking value
 *   for the flow. otherwise status->proto_state will be initialized with zeros.
 * @param dev_id
 *   Event dev ID to enqueue end of flow event.
 * @param port_id
 *   Event port ID to enqueue end of flow event.
 * @param action_specs
 *   Hold the actions configuration.
 * @param[out] mbuf_out
 *   last processed not fragmented and in order mbuf.
 * @param[out] status
 *   Structure to dump SFT flow status once activated.
 * @param[out] error
 *   Perform verbose error reporting if not NULL. SFT initialize this
 *   structure in case of error only.
 *
 * @return
 *   0 on success, a negative errno value otherwise and rte_sft_error is set.
 */
__rte_experimental
int
rte_sft_flow_activate(uint16_t queue, uint32_t zone, struct rte_mbuf *mbuf_in,
		      const struct rte_sft_7tuple *reverse_tuple,
		      uint8_t state, uint32_t *data, uint8_t proto_enable,
		      const struct rte_sft_actions_specs *action_specs,
		      uint8_t dev_id, uint8_t port_id,
		      struct rte_mbuf **mbuf_out,
		      struct rte_sft_flow_status *status,
		      struct rte_sft_error *error);

/**
 * @warning
 * @b EXPERIMENTAL: this API may change without prior notice.
 *
 * Artificially create SFT flow.
 *
 * Function to create SFT flow before reception of the first flow packet.
 *
 * @param queue
 *   The SFT queue.
 * @param tuple
 *   Expected initiator flow 7-tuple.
 * @param reverse_tuple
 *   Expected initiate flow 7-tuple.
 * @param ctx
 *   SFT flow item context.
 * @param ct_enable
 *   Connection tracking enable.
 * @param[out] status
 *   Connection status.
 * @param[out] error
 *   Perform verbose error reporting if not NULL. PMDs initialize this
 *   structure in case of error only.
 *
 * @return
 *   - on success: 0, locked SFT flow recognized by status->fid.
 *   - on error: a negative errno value otherwise and rte_errno is set.
 */
__rte_experimental
int
rte_sft_flow_create(uint16_t queue, const struct rte_sft_7tuple *tuple,
		    const struct rte_sft_7tuple *reverse_tuple,
		    const struct rte_flow_item_sft *ctx,
		    uint8_t ct_enable,
		    struct rte_sft_flow_status *status,
		    struct rte_sft_error *error);

/**
 * @warning
 * @b EXPERIMENTAL: this API may change without prior notice.
 *
 * Removes flow from SFT.
 *
 * @param queue
 *   The SFT queue.
 * @param fid
 *   SFT flow ID to destroy.
 * @param[out] error
 *   Perform verbose error reporting if not NULL. SFT initialize this
 *   structure in case of error only.
 *
 * @return
 *   0 on success, a negative errno value otherwise and rte_sft_error is set.
 */
__rte_experimental
int
rte_sft_flow_destroy(uint16_t queue, uint32_t fid, struct rte_sft_error *error);

/**
 * @warning
 * @b EXPERIMENTAL: this API may change without prior notice.
 *
 * Query counter and aging data.
 *
 * @param queue
 *   The SFT queue.
 * @param fid
 *   SFT flow ID.
 * @param[out] data
 *   SFT flow ID.
 * @param[out] error
 *   Perform verbose error reporting if not NULL. SFT initialize this
 *   structure in case of error only.
 *
 * @return
 *   0 on success, a negative errno value otherwise and rte_sft_error is set.
 */
__rte_experimental
int
rte_sft_flow_query(uint16_t queue, uint32_t fid,
		   struct rte_sft_query_data *data,
		   struct rte_sft_error *error);

/**
 * @warning
 * @b EXPERIMENTAL: this API may change without prior notice.
 *
 * Reset flow age to zero.
 *
 * Simulates last flow packet with timestamp set to just now.
 *
 * @param queue
 *   The SFT queue.
 * @param fid
 *   SFT flow ID.
 * @param[out] error
 *   Perform verbose error reporting if not NULL. SFT initialize this
 *   structure in case of error only.
 *
 * @return
 *   0 on success, a negative errno value otherwise and rte_sft_error is set.
 */
__rte_experimental
int
rte_sft_flow_touch(uint16_t queue, uint32_t fid, struct rte_sft_error *error);

/**
 * @warning
 * @b EXPERIMENTAL: this API may change without prior notice.
 *
 * Set flow aging to specific value.
 *
 * @param queue
 *   The SFT queue.
 * @param fid
 *   SFT flow ID.
 * @param aging
 *   New flow aging value.
 * @param[out] error
 *   Perform verbose error reporting if not NULL. SFT initialize this
 *   structure in case of error only.
 *
 * @return
 *   0 on success, a negative errno value otherwise and rte_sft_error is set.
 */
__rte_experimental
int
rte_sft_flow_set_aging(uint16_t queue, uint32_t fid, uint32_t aging,
		       struct rte_sft_error *error);

/**
 * @warning
 * @b EXPERIMENTAL: this API may change without prior notice.
 *
 * Set client object for given client ID.
 *
 * @param queue
 *   The SFT queue.
 * @param fid
 *   SFT flow ID.
 * @param client_id
 *   Client ID to set object for.
 * @param client_obj
 *   Pointer to opaque client object structure.
 * @param[out] error
 *   Perform verbose error reporting if not NULL. SFT initialize this
 *   structure in case of error only.
 *
 * @return
 *   0 on success, a negative errno value otherwise and rte_sft_error is set.
 */
__rte_experimental
int
rte_sft_flow_set_client_obj(uint16_t queue, uint32_t fid, uint8_t client_id,
			    const void *client_obj,
			    struct rte_sft_error *error);

/**
 * @warning
 * @b EXPERIMENTAL: this API may change without prior notice.
 *
 * Get client object for given client ID.
 *
 * @param queue
 *   The SFT queue.
 * @param fid
 *   SFT flow ID.
 * @param client_id
 *   Client ID to get object for.
 * @param[out] error
 *   Perform verbose error reporting if not NULL. SFT initialize this
 *   structure in case of error only.
 *
 * @return
 *   A valid client object opaque pointer in case of success, NULL otherwise
 *   and rte_sft_error is set.
 */
__rte_experimental
void *
rte_sft_flow_get_client_obj(uint16_t queue, const uint32_t fid,
			    uint8_t client_id, struct rte_sft_error *error);

/**
 * @warning
 * @b EXPERIMENTAL: this API may change without prior notice.
 *
 * Drain IP fragments after all data arrived.
 */
__rte_experimental
int
rte_sft_drain_fragment_mbuf(uint16_t queue, uint32_t zone, uintptr_t frag_ctx,
			    uint16_t num_to_drain, struct rte_mbuf **mbuf_out,
			    struct rte_sft_flow_status *status,
			    struct rte_sft_error *error);

/**
 * @warning
 * @b EXPERIMENTAL: this API may change without prior notice.
 *
 * Set SFT error details.
 */
int
rte_sft_error_set(struct rte_sft_error *error,
		  int code,
		  enum rte_sft_error_type type,
		  const void *cause,
		  const char *message);

__rte_experimental
void
rte_sft_debug(uint16_t port_id, uint16_t queue, uint32_t fid,
	      struct rte_sft_error *error);

/**
 * context for SFT mbuf parser
 */
__extension__
struct rte_sft_mbuf_info {
	const struct rte_ether_hdr *eth_hdr;
	union {
		const struct rte_ipv4_hdr *ip4;
		const struct rte_ipv6_hdr *ip6;
		const void *l3_hdr;
	};
	const struct ipv6_extension_fragment *ip6_frag;
	union {
		const struct rte_tcp_hdr *tcp;
		const struct rte_udp_hdr *udp;
		const void *l4_hdr;
	};
	uint16_t eth_type;
	uint16_t data_len;
	uint32_t l4_protocol:8;
	uint32_t is_fragment:1;
	uint32_t is_ipv6:1;
	uint32_t direction_located:1;
};

/**
 * @warning
 * @b EXPERIMENTAL: this API may change without prior notice.
 *
 * SFT mbuf parser.
 */
__rte_experimental
int
rte_sft_parse_mbuf(const struct rte_mbuf *m, struct rte_sft_mbuf_info *mif,
		   const void *entry, struct rte_sft_error *error);

/**
 * @warning
 * @b EXPERIMENTAL: this API may change without prior notice.
 *
 * Fill in SFT 7-tuple.
 */
__rte_experimental
void
rte_sft_mbuf_stpl(const struct rte_mbuf *m, struct rte_sft_mbuf_info *mif,
		  uint32_t zone, struct rte_sft_7tuple *stpl,
		  struct rte_sft_error *error);

extern int sft_logtype;
#define RTE_SFT_LOG(level, ...) \
	rte_log(RTE_LOG_ ## level, sft_logtype, "" __VA_ARGS__)

#ifdef __cplusplus
}
#endif

#endif /* _RTE_SFT_H_ */
