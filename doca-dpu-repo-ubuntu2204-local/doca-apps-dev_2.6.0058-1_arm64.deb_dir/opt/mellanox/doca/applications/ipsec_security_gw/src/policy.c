/*
 * Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES, ALL RIGHTS RESERVED.
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

#include <netinet/in.h>
#include <time.h>

#include <doca_argp.h>
#include <doca_log.h>
#include <doca_flow.h>

#include <samples/common.h>
#include <flow_parser.h>

#include "policy.h"
#include "config.h"

DOCA_LOG_REGISTER(IPSEC_SECURITY_GW::POLICY);

/*
 * Convert ICV length to doca_ipsec_icv_length value
 *
 * @icv_length [in]: ICV length {8, 12, 16}
 * @length [out]: suitable doca_ipsec_icv_length value
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
static doca_error_t
convert_to_doca_icv(uint8_t icv_length, enum doca_ipsec_icv_length *length)
{
	doca_error_t result = DOCA_SUCCESS;

	switch (icv_length) {
	case 8:
		*length = DOCA_IPSEC_ICV_LENGTH_8;
		break;
	case 12:
		*length = DOCA_IPSEC_ICV_LENGTH_12;
		break;
	case 16:
		*length = DOCA_IPSEC_ICV_LENGTH_16;
		break;
	default:
		result = DOCA_ERROR_NOT_SUPPORTED;
	}
	return result;
}

/*
 * Convert key type to doca_encryption_key_type value
 *
 * @key_type [in]: key type {128, 256}
 * @return: suitable doca_encryption_key_type value
 */
static enum doca_encryption_key_type
convert_to_doca_key_type(uint8_t key_type)
{
	if (key_type == POLICY_KEY_TYPE_128)
		return DOCA_ENCRYPTION_KEY_AESGCM_128;
	else
		return DOCA_ENCRYPTION_KEY_AESGCM_256;
}

/*
 * Parse SA attributes
 *
 * @policy [in]: application IPSEC policy
 * @sa_attrs [out]: SA app structure
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
static doca_error_t
parse_sa_attrs(struct ipsec_security_gw_ipsec_policy *policy, struct ipsec_security_gw_sa_attrs *sa_attrs)
{
	enum doca_ipsec_icv_length icv_length;
	doca_error_t result;

	result = convert_to_doca_icv(policy->icv_length, &icv_length);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to convert ICV length");
		return result;
	}

	sa_attrs->key_type = convert_to_doca_key_type(policy->key_type);
	memcpy(sa_attrs->enc_key_data, policy->enc_key_data, MAX_KEY_LEN);
	sa_attrs->salt = policy->salt;
	sa_attrs->icv_length = icv_length;

	return DOCA_SUCCESS;
}

/*
 * Parse new ingress policy and populate the encryption rule structure
 *
 * @policy [in]: application IPSEC policy
 * @rule [out]: encryption rule structure
 * @ip6_table [out]: store hash value for IPV6 addresses
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
static doca_error_t
ipsec_security_gw_policy_encrypt_parse(struct ipsec_security_gw_ipsec_policy *policy, struct encrypt_rule *rule,
					struct rte_hash **ip6_table)
{
	doca_error_t result;
	int ret;

	rule->esp_spi = policy->spi;
	rule->l3_type = (policy->l3_protocol == POLICY_L3_TYPE_IPV4) ? DOCA_FLOW_L3_TYPE_IP4 : DOCA_FLOW_L3_TYPE_IP6;
	rule->protocol = (policy->l4_protocol == POLICY_L4_TYPE_UDP) ? DOCA_FLOW_L4_TYPE_EXT_UDP : DOCA_FLOW_L4_TYPE_EXT_TCP;
	rule->src_port = policy->src_port;
	rule->dst_port = policy->dst_port;

	if (rule->l3_type == DOCA_FLOW_L3_TYPE_IP4) {
		result = parse_ipv4_str(&policy->src_ip_addr[0], &rule->ip4.src_ip);
		if (result != DOCA_SUCCESS)
			return result;

		result = parse_ipv4_str(&policy->dst_ip_addr[0], &rule->ip4.dst_ip);
		if (result != DOCA_SUCCESS)
			return result;
	} else {
		result = parse_ipv6_str(&policy->src_ip_addr[0], rule->ip6.src_ip);
		if (result != DOCA_SUCCESS)
			return result;

		/* Add IPV6 source IP address to hash table */
		ret = rte_hash_lookup(*ip6_table, (void *)rule->ip6.src_ip);
		if (ret < 0) {
			ret = rte_hash_add_key(*ip6_table, rule->ip6.src_ip);
			if (ret < 0) {
				DOCA_LOG_ERR("Failed to add address to hash table");
				return DOCA_ERROR_DRIVER;
			}
		}

		result = parse_ipv6_str(&policy->dst_ip_addr[0], rule->ip6.dst_ip);
		if (result != DOCA_SUCCESS)
			return result;
		ret = rte_hash_lookup(*ip6_table, (void *)rule->ip6.dst_ip);
		if (ret < 0) {
			ret = rte_hash_add_key(*ip6_table, rule->ip6.dst_ip);
			if (ret < 0) {
				DOCA_LOG_ERR("Failed to add address to hash table");
				return DOCA_ERROR_DRIVER;
			}
		}
	}

	/* If policy mode is tunnel, parse the outer header attributes  */
	if (policy->policy_mode == POLICY_MODE_TUNNEL) {
		rule->encap_l3_type = (policy->outer_l3_protocol == POLICY_L3_TYPE_IPV4) ? DOCA_FLOW_L3_TYPE_IP4 : DOCA_FLOW_L3_TYPE_IP6;
		if (rule->encap_l3_type == DOCA_FLOW_L3_TYPE_IP4) {
			result = parse_ipv4_str(&policy->outer_dst_ip[0], &rule->encap_dst_ip4);
			if (result != DOCA_SUCCESS)
				return result;
		} else {
			result = parse_ipv6_str(&policy->outer_dst_ip[0], rule->encap_dst_ip6);
			if (result != DOCA_SUCCESS)
				return result;
		}
	}

	result = parse_sa_attrs(policy, &rule->sa_attrs);
	if (result != DOCA_SUCCESS)
		return result;

	rule->sa_attrs.direction = DOCA_IPSEC_DIRECTION_EGRESS_ENCRYPT;

	return DOCA_SUCCESS;
}

/*
 * Parse new egress policy and populate the decryption rule structure
 *
 * @policy [in]: application IPSEC policy
 * @mode [in]: application IPSEC mode
 * @rule [out]: decryption rule structure
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
static doca_error_t
ipsec_security_gw_policy_decrypt_parse(struct ipsec_security_gw_ipsec_policy *policy, enum ipsec_security_gw_mode mode,
					struct decrypt_rule *rule)
{
	enum doca_flow_l3_type outer_l3_type = (policy->outer_l3_protocol == POLICY_L3_TYPE_IPV4) ? DOCA_FLOW_L3_TYPE_IP4 : DOCA_FLOW_L3_TYPE_IP6;
	enum doca_flow_l3_type inner_l3_type = (policy->l3_protocol == POLICY_L3_TYPE_IPV4) ? DOCA_FLOW_L3_TYPE_IP4 : DOCA_FLOW_L3_TYPE_IP6;
	char *dst_ip = NULL;
	doca_error_t result;

	rule->inner_l3_type = inner_l3_type;

	if (mode == IPSEC_SECURITY_GW_TUNNEL) {
		rule->l3_type = outer_l3_type;
		dst_ip = &policy->outer_dst_ip[0];

	} else {
		rule->l3_type = inner_l3_type;
		dst_ip = &policy->dst_ip_addr[0];
	}

	if (rule->l3_type == DOCA_FLOW_L3_TYPE_IP4) {
		result = parse_ipv4_str(dst_ip, &rule->dst_ip4);
		if (result != DOCA_SUCCESS)
			return result;
	} else {
		result = parse_ipv6_str(dst_ip, rule->dst_ip6);
		if (result != DOCA_SUCCESS)
			return result;
	}

	rule->esp_spi = policy->spi;

	result = parse_sa_attrs(policy, &rule->sa_attrs);
	if (result != DOCA_SUCCESS)
		return result;

	rule->sa_attrs.direction = DOCA_IPSEC_DIRECTION_INGRESS_DECRYPT;

	return DOCA_SUCCESS;
}

doca_error_t
ipsec_security_gw_handle_encrypt_policy(struct ipsec_security_gw_config *app_cfg,
	struct ipsec_security_gw_ports_map *ports[], struct ipsec_security_gw_ipsec_policy *policy, struct encrypt_rule *rule)
{
	doca_error_t result;

	result = ipsec_security_gw_policy_encrypt_parse(policy, rule, &app_cfg->ip6_table);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to parse new encryption policy");
		return result;
	}

	result = ipsec_security_gw_create_ipsec_sa(&rule->sa_attrs, app_cfg, &rule->sa);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to create new SA for new encryption policy");
		return result;
	}

	result = add_encrypt_entry(rule, app_cfg->app_rules.nb_rules, ports, app_cfg);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to insert entries for encryption policy");
		return result;
	}

	app_cfg->app_rules.nb_encrypted_rules++;
	app_cfg->app_rules.nb_rules++;

	return DOCA_SUCCESS;
}

doca_error_t
ipsec_security_gw_handle_decrypt_policy(struct ipsec_security_gw_config *app_cfg, struct doca_flow_port *secured_port,
					struct ipsec_security_gw_ipsec_policy *policy, struct decrypt_rule *rule)
{
	doca_error_t result;

	result = ipsec_security_gw_policy_decrypt_parse(policy, app_cfg->mode, rule);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to parse new decryption policy");
		return result;
	}

	result = ipsec_security_gw_create_ipsec_sa(&rule->sa_attrs, app_cfg, &rule->sa);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to create new SA for new decryption policy");
		return result;
	}

	result = add_decrypt_entry(rule, app_cfg->app_rules.nb_rules, secured_port, app_cfg);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to insert entries for decryption policy");
		return result;
	}

	app_cfg->app_rules.nb_decrypted_rules++;
	app_cfg->app_rules.nb_rules++;
	return DOCA_SUCCESS;
}

void
print_policy_attrs(struct ipsec_security_gw_ipsec_policy *policy)
{
	char *dump;
	int log_level;
	uint16_t key_len;

	if (doca_argp_get_log_level(&log_level) != DOCA_SUCCESS)
		return;

	/* Logs below should be in debug mode, having a check here will save time of calling multiple
	 * DOCA log prints and hex_dump function.
	 */
	if (log_level != DOCA_LOG_LEVEL_DEBUG)
		return;

	/* Set key length in bytes */
	key_len = (policy->key_type == POLICY_KEY_TYPE_128) ? 16 : 32;

	DOCA_LOG_DBG("Policy L3 protocol %u == %s", policy->l3_protocol, (policy->l3_protocol == POLICY_L3_TYPE_IPV4) ? "IPV4" : "IPV6");
	DOCA_LOG_DBG("Policy l4_protocol %u == %s", policy->l4_protocol, (policy->l4_protocol == IPPROTO_UDP) ? "UDP" : "TCP");
	DOCA_LOG_DBG("Policy src_ip_addr %s", policy->src_ip_addr);
	DOCA_LOG_DBG("Policy dst_ip_addr %s", policy->dst_ip_addr);
	DOCA_LOG_DBG("Policy src_port %u", policy->src_port);
	DOCA_LOG_DBG("Policy dst_port %u", policy->dst_port);
	DOCA_LOG_DBG("Policy policy_direction %u == %s", policy->policy_direction, (policy->policy_direction == POLICY_DIR_OUT) ? "OUT" : "IN");
	DOCA_LOG_DBG("Policy policy_mode %u == %s", policy->policy_mode, (policy->policy_mode == POLICY_MODE_TUNNEL) ? "TUNNEL" : "TRANSPORT");
	DOCA_LOG_DBG("Policy spi 0x%x", policy->spi);
	dump = hex_dump(policy->enc_key_data, key_len);
	DOCA_LOG_DBG("Policy enc_key_data =\n%s", dump);
	free(dump);
	DOCA_LOG_DBG("Policy salt %u", policy->salt);
	DOCA_LOG_DBG("Policy key_type %u", (policy->key_type == POLICY_KEY_TYPE_128) ? 128 : 256);
	DOCA_LOG_DBG("Policy esn %u", policy->esn);
	DOCA_LOG_DBG("Policy icv_length %u", policy->icv_length);
	DOCA_LOG_DBG("Policy outer_src_ip %s", policy->outer_src_ip);
	DOCA_LOG_DBG("Policy outer_dst_ip %s", policy->outer_dst_ip);
	DOCA_LOG_DBG("Policy outer_l3_protocol %s", (policy->outer_l3_protocol == POLICY_L3_TYPE_IPV4) ? "IPV4" : "IPV6");
}
