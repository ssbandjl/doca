/*
 * Copyright (c) 2017-2020 Mellanox Technologies LTD. All rights reserved.
 * Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * This software is available to you under the terms of the
 * OpenIB.org BSD license included below:
 *
 *     Redistribution and use in source and binary forms, with or
 *     without modification, are permitted provided that the following
 *     conditions are met:
 *
 *      - Redistributions of source code must retain the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer.
 *
 *      - Redistributions in binary form must reproduce the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer in the documentation and/or other materials
 *        provided with the distribution.
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

#ifndef IBDIAG_FABRIC_H
#define IBDIAG_FABRIC_H

#include <stdlib.h>
#include <infiniband/ibdm/Fabric.h>
#include "ibdiag_ibdm_extended_info.h"
#include <ibis/ibis.h>

class NodeRecord;
class PortRecord;
class SwitchRecord;
class LinkRecord;
class GeneralInfoSMPRecord;
class GeneralInfoGMPRecord;
class ExtendedNodeInfoRecord;
class ExtendedPortInfoRecord;
class PortInfoExtendedRecord;
class PortHierarchyInfoRecord;
class PhysicalHierarchyInfoRecord;
class ARInfoRecord;

class IBDiagFabric {
private:

    CsvParser m_csv_parser;

    IBFabric &discovered_fabric;
    IBDMExtendedInfo &fabric_extended_info;
    CapabilityModule &capability_module;;

    u_int32_t nodes_found;
    u_int32_t sw_found;
    u_int32_t ca_found;
    u_int64_t ports_found;

    string    last_error;

    int CreateDummyPorts();

public:

    const string& GetLastError() const { return last_error; }

    u_int32_t getNodesFound() { return nodes_found;}
    u_int32_t getSWFound()    { return sw_found;}
    u_int32_t getCAFound()    { return ca_found;}
    u_int64_t getPortsFound() { return ports_found;}

    int UpdateFabric(const string &csv_file);

    int CreateNode(const NodeRecord &nodeRecord);
    int CreatePort(const PortRecord &portRecord);
    int CreateSwitch(const SwitchRecord &switchRecord);
    int CreateLink(const LinkRecord &linkRecord);
    int CreateVSGeneralInfoSMP(const GeneralInfoSMPRecord &generalInfoSMPRecord);
    int CreateVSGeneralInfoGMP(const GeneralInfoGMPRecord &generalInfoGMPRecord);
    int CreateExtendedNodeInfo(const ExtendedNodeInfoRecord &extendedNodeInfoRecord);
    int CreateExtendedPortInfo(const ExtendedPortInfoRecord &extendedPortInfoRecord);
    int CreatePortInfoExtended(const PortInfoExtendedRecord &portInfoExtendedRecord);
    int CreatePortHierarchyInfo(const PortHierarchyInfoRecord &portHierarchyInfoRecord);
    int CreatePhysicalHierarchyInfo(const PhysicalHierarchyInfoRecord &physicalHierarchyInfoRecord);
    int CreateARInfo(const ARInfoRecord &arInfoRecord);

    IBDiagFabric(IBFabric &discovered_fabric, IBDMExtendedInfo &fabric_extended_info,
                 CapabilityModule &capability_module) :
        discovered_fabric(discovered_fabric), fabric_extended_info(fabric_extended_info),
        capability_module(capability_module), nodes_found(0), sw_found(0), ca_found(0),
        ports_found(0) { }
};


/*******************************
 *       SECTION RECORDS       *
 *******************************/
class NodeRecord {

public:

    string              node_description;
    u_int16_t           num_ports;
    u_int8_t            node_type;
    u_int8_t            class_version;
    u_int8_t            base_version;
    u_int64_t           system_image_guid;
    u_int64_t           node_guid;
    u_int64_t           port_guid;
    u_int16_t           device_id;
    u_int16_t           partition_cap;
    u_int32_t           revision;
    u_int32_t           vendor_id;
    u_int8_t            local_port_num;

    NodeRecord()
        : num_ports(0), node_type(0), class_version(0), base_version(0),
          system_image_guid(0), node_guid(0), port_guid(0), device_id(0),
          partition_cap(0), revision(0), vendor_id(0), local_port_num(0)
    {}

    static int Init(vector < ParseFieldInfo <class NodeRecord> > &parse_section_info);

    bool SetNodeDescription(const char *field_str) {
        return CsvParser::Parse(field_str, node_description);
    }

    bool SetNumPorts(const char *field_str) {
        return CsvParser::Parse(field_str, num_ports);
    }

    bool SetNodeType(const char *field_str) {
        return CsvParser::Parse(field_str, node_type);
    }

    bool SetClassVersion(const char *field_str) {
        return CsvParser::Parse(field_str, class_version);
    }

    bool SetBaseVersion(const char *field_str) {
        return CsvParser::Parse(field_str, base_version);
    }

    bool SetSystemImageGUID(const char *field_str) {
        return CsvParser::Parse(field_str, system_image_guid, 16);
    }

    bool SetNodeGUID(const char *field_str) {
        return CsvParser::Parse(field_str, node_guid, 16);
    }

    bool SetPortGUID(const char *field_str) {
        return CsvParser::Parse(field_str, port_guid, 16);
    }

    bool SetDeviceID(const char *field_str) {
        return CsvParser::Parse(field_str, device_id);
    }

    bool SetPartitionCap(const char *field_str) {
        return CsvParser::Parse(field_str, partition_cap);
    }

    bool SetRevision(const char *field_str) {
        return CsvParser::Parse(field_str, revision);
    }

    bool SetVendorID(const char *field_str) {
        return CsvParser::Parse(field_str, vendor_id);
    }

    bool SetLocalPortNum(const char *field_str) {
        return CsvParser::Parse(field_str, local_port_num);
    }
};

class PortRecord {

public:
// 1
    u_int64_t           node_guid;
    u_int64_t           port_guid;
    u_int8_t            port_num;
    u_int64_t           m_key;
    u_int64_t           gid_prefix;
    u_int16_t           msm_lid;
    u_int16_t           lid;
// 2
    u_int32_t           cap_mask;
    u_int16_t           m_key_lease_period;
    u_int16_t           diag_code;
    u_int8_t            link_width_actv;
    u_int8_t            link_width_sup;
    u_int8_t            link_width_en;
    u_int8_t            local_port_num;
// 3
    u_int32_t           link_speed_en;
    u_int32_t           link_speed_actv;
    u_int8_t            lmc;
    u_int8_t            m_key_prot_bits;
    u_int8_t            link_down_def_state;
    u_int8_t            port_phy_state;
    u_int8_t            port_state;
    u_int32_t           link_speed_sup;
// 4
    u_int8_t            vl_arbit_High_Cap;
    u_int8_t            vl_high_limit;
    u_int8_t            init_type;
    u_int8_t            vl_cap;
    u_int8_t            msm_sl;
    u_int8_t            nmtu;
    u_int8_t            filter_raw_outbound;
// 5
    u_int8_t            filter_raw_inbound;
    u_int8_t            part_enf_outbound;
    u_int8_t            part_enf_inbound;
    u_int8_t            op_VLs;
    u_int8_t            hoq_life;
    u_int8_t            vl_stall_cnt;
    u_int8_t            mtu_cap;
// 6
    u_int8_t            init_type_reply;
    u_int8_t            vl_arbit_low_cap;
    u_int16_t           pkey_violations;
    u_int16_t           mkey_violations;
    u_int8_t            subn_time_out;
    u_int8_t            multicast_pkey_trap_suppression_enabled;
    u_int8_t            client_reregister;
    u_int8_t            guid_cap;
// 7
    u_int16_t           qkey_violations;
    u_int16_t           max_credit_hint;
    u_int8_t            overrun_errs;
    u_int8_t            local_phy_error;
    u_int8_t            resp_time_value;
    u_int32_t           link_round_trip_latency;
// 8
    u_int16_t           cap_mask_2;

    // TODO::remove from db_csv ???
    string              fec_actv;
    string              retrans_actv;

    PortRecord()
        :node_guid(0), port_guid(0), port_num(0), m_key(0), gid_prefix(0),
         msm_lid(0), lid(0), cap_mask(0), m_key_lease_period(0),
         diag_code(0), link_width_actv(0), link_width_sup(0), link_width_en(0),
         local_port_num(0), link_speed_en(0), link_speed_actv(0), lmc(0),
         m_key_prot_bits(0), link_down_def_state(0), port_phy_state(0), port_state(0),
         link_speed_sup(0), vl_arbit_High_Cap(0), vl_high_limit(0), init_type(0),
         vl_cap(0), msm_sl(0), nmtu(0), filter_raw_outbound(0),
         filter_raw_inbound(0), part_enf_outbound(0), part_enf_inbound(0), op_VLs(0),
         hoq_life(0), vl_stall_cnt(0), mtu_cap(0), init_type_reply(0), vl_arbit_low_cap(0),
         pkey_violations(0), mkey_violations(0), subn_time_out(0),
         multicast_pkey_trap_suppression_enabled(0), client_reregister(0), guid_cap(0),
         qkey_violations(0), max_credit_hint(0), overrun_errs(0), local_phy_error(0),
         resp_time_value(0), link_round_trip_latency(0), cap_mask_2(0)
    {}

    static int Init(vector < ParseFieldInfo <class PortRecord> > &parse_section_info);

//  *** 1
    bool SetNodeGuid(const char *field_str) {
        return CsvParser::Parse(field_str, node_guid, 16);
    }

    bool SetPortGuid(const char *field_str) {
        return CsvParser::Parse(field_str, port_guid, 16);
    }

    bool SetPortNum(const char *field_str) {
        return CsvParser::Parse(field_str, port_num);
    }

    bool SetMKey(const char *field_str) {
        return CsvParser::Parse(field_str, m_key, 16);
    }

    bool SetGIDPrfx(const char *field_str) {
        return CsvParser::Parse(field_str, gid_prefix, 16);
    }

    bool SetMSMLID(const char *field_str) {
        return CsvParser::Parse(field_str, msm_lid);
    }

    bool SetLid(const char *field_str) {
        return CsvParser::Parse(field_str, lid);
    }

//  *** 2
    bool SetCapMsk(const char *field_str) {
        return CsvParser::Parse(field_str, cap_mask);
    }

    bool SetM_KeyLeasePeriod(const char *field_str) {
        return CsvParser::Parse(field_str, m_key_lease_period);
    }

    bool SetDiagCode(const char *field_str) {
        return CsvParser::Parse(field_str, diag_code);
    }

    bool SetLinkWidthActv(const char *field_str) {
        return CsvParser::Parse(field_str, link_width_actv);
    }

    bool SetLinkWidthSup(const char *field_str) {
        return CsvParser::Parse(field_str, link_width_sup);
    }

    bool SetLinkWidthEn(const char *field_str) {
        return CsvParser::Parse(field_str, link_width_en);
    }

    bool SetLocalPortNum(const char *field_str) {
        return CsvParser::Parse(field_str, local_port_num);
    }

//  *** 3
    bool SetLinkSpeedEn(const char *field_str) {
        return CsvParser::Parse(field_str, link_speed_en);
    }

    bool SetLinkSpeedActv(const char *field_str) {
        return CsvParser::Parse(field_str, link_speed_actv);
    }

    bool SetLMC(const char *field_str) {
        return CsvParser::Parse(field_str, lmc);
    }

    bool SetMKeyProtBits(const char *field_str) {
        return CsvParser::Parse(field_str, m_key_prot_bits);
    }

    bool SetLinkDownDefState(const char *field_str) {
        return CsvParser::Parse(field_str, link_down_def_state);
    }

    bool SetPortPhyState(const char *field_str) {
        return CsvParser::Parse(field_str, port_phy_state);
    }

    bool SetPortState(const char *field_str) {
        return CsvParser::Parse(field_str, port_state);
    }

    bool SetLinkSpeedSup(const char *field_str) {
        return CsvParser::Parse(field_str, link_speed_sup);
    }

//  *** 4
    bool SetVLArbHighCap(const char *field_str) {
        return CsvParser::Parse(field_str, vl_arbit_High_Cap);
    }

    bool SetVLHighLimit(const char *field_str) {
        return CsvParser::Parse(field_str, vl_high_limit);
    }

    bool SetInitType(const char *field_str) {
        return CsvParser::Parse(field_str, init_type);
    }

    bool SetVLCap(const char *field_str) {
        return CsvParser::Parse(field_str, vl_cap);
    }

    bool SetMSMSL(const char *field_str) {
        return CsvParser::Parse(field_str, msm_sl);
    }

    bool SetNMTU(const char *field_str) {
        return CsvParser::Parse(field_str, nmtu);
    }

    bool SetFilterRawOutb(const char *field_str) {
        return CsvParser::Parse(field_str, filter_raw_outbound);
    }

//  *** 5
    bool SetFilterRawInb(const char *field_str) {
        return CsvParser::Parse(field_str, filter_raw_inbound);
    }

    bool SetPartEnfOutb(const char *field_str) {
        return CsvParser::Parse(field_str, part_enf_outbound);
    }

    bool SetPartEnfInb(const char *field_str) {
        return CsvParser::Parse(field_str, part_enf_inbound);
    }

    bool SetOpVLs(const char *field_str) {
        return CsvParser::Parse(field_str, op_VLs);
    }

    bool SetHoQLife(const char *field_str) {
        return CsvParser::Parse(field_str, hoq_life);
    }

    bool SetVLStallCnt(const char *field_str) {
        return CsvParser::Parse(field_str, vl_stall_cnt);
    }

    bool SetMTUCap(const char *field_str) {
        return CsvParser::Parse(field_str, mtu_cap);
    }

//  *** 6
    bool SetInitTypeReply(const char *field_str) {
        return CsvParser::Parse(field_str, init_type_reply);
    }

    bool SetVLArbLowCap(const char *field_str) {
        return CsvParser::Parse(field_str, vl_arbit_low_cap);
    }

    bool SetPKeyViolations(const char *field_str) {
        return CsvParser::Parse(field_str, pkey_violations);
    }

    bool SetMKeyViolations(const char *field_str) {
        return CsvParser::Parse(field_str, mkey_violations);
    }

    bool SetSubnTmo(const char *field_str) {
        return CsvParser::Parse(field_str, subn_time_out);
    }

    bool SetClientReregister(const char *field_str) {
        return CsvParser::Parse(field_str, client_reregister);
    }

    bool SetMulticastPKeyTrapSuppressionEnabled(const char *field_str) {
        return CsvParser::Parse(field_str, multicast_pkey_trap_suppression_enabled);
    }

    bool SetGUIDCap(const char *field_str) {
        return CsvParser::Parse(field_str, guid_cap);
    }

//  *** 7
    bool SetQKeyViolations(const char *field_str) {
        return CsvParser::Parse(field_str, qkey_violations);
    }

    bool SetMaxCreditHint(const char *field_str) {
        return CsvParser::Parse(field_str, max_credit_hint);
    }

    bool SetOverrunErrs(const char *field_str) {
        return CsvParser::Parse(field_str, overrun_errs);
    }

    bool SetLocalPhyError(const char *field_str) {
        return CsvParser::Parse(field_str, local_phy_error);
    }

    bool SetRespTimeValue(const char *field_str) {
        return CsvParser::Parse(field_str, resp_time_value);
    }

    bool SetLinkRoundTripLatency(const char *field_str) {
        return CsvParser::Parse(field_str, link_round_trip_latency);
    }

//  *** 8
    bool SetCapMsk2(const char *field_str) {
        return CsvParser::ParseWithNA(field_str, cap_mask_2);
    }

    bool SetFECActv(const char *field_str) {
        return CsvParser::Parse(field_str, fec_actv);
    }

    bool SetRetransActv(const char *field_str) {
        return CsvParser::Parse(field_str, retrans_actv);
    }
};

class SwitchRecord {

public:
    u_int64_t           node_guid;

    u_int16_t           linear_FDB_cap;
    u_int16_t           random_FDB_cap;
    u_int16_t           mcast_FDB_cap;
    u_int16_t           linear_FDB_top;

    u_int8_t            def_port;
    u_int8_t            def_mcast_pri_port;
    u_int8_t            def_mcast_not_pri_port;
    u_int8_t            life_time_value;

    u_int8_t            port_state_change;
    u_int8_t            optimized_SLVL_mapping;
    u_int16_t           lids_per_port;
    u_int16_t           part_enf_cap;

    u_int8_t            inb_enf_cap;
    u_int8_t            outb_enf_cap;
    u_int8_t            filter_raw_inb_cap;
    u_int8_t            filter_raw_outb_cap;

    u_int8_t            en_port0;
    u_int16_t           mcast_FDB_top;

    SwitchRecord()
        : node_guid(0),
          linear_FDB_cap(0), random_FDB_cap(0), mcast_FDB_cap(0), linear_FDB_top(0),
          def_port(0), def_mcast_pri_port(0), def_mcast_not_pri_port(0), life_time_value(0),
          port_state_change(0), optimized_SLVL_mapping(0),
          lids_per_port(0), part_enf_cap(0),inb_enf_cap(0),outb_enf_cap(0),
          filter_raw_inb_cap(0),filter_raw_outb_cap(0),
          en_port0(0), mcast_FDB_top(0)

    {}

    static int Init(vector < ParseFieldInfo <class SwitchRecord> > &parse_section_info);

    bool SetNodeGuid(const char *field_str) {
        return CsvParser::Parse(field_str, node_guid, 16);
    }

    bool SetLinearFDBCap(const char *field_str) {
       return CsvParser::Parse(field_str, linear_FDB_cap);
    }

    bool SetRandomFDBCap(const char *field_str) {
       return CsvParser::Parse(field_str, random_FDB_cap);
    }

    bool SetMCastFDBCap(const char *field_str) {
       return CsvParser::Parse(field_str, mcast_FDB_cap);
    }

    bool SetLinearFDBTop(const char *field_str) {
       return CsvParser::Parse(field_str, linear_FDB_top);
    }

    bool SetDefPort(const char *field_str) {
       return CsvParser::Parse(field_str, def_port);
    }

    bool SetDefMCastPriPort(const char *field_str) {
       return CsvParser::Parse(field_str, def_mcast_pri_port);
    }

    bool SetDefMCastNotPriPort(const char *field_str) {
       return CsvParser::Parse(field_str, def_mcast_not_pri_port);
    }

    bool SetLifeTimeValue(const char *field_str) {
       return CsvParser::Parse(field_str, life_time_value);
    }

    bool SetPortStateChange(const char *field_str) {
       return CsvParser::Parse(field_str, port_state_change);
    }

    bool SetOptimizedSLVLMapping(const char *field_str) {
       return CsvParser::Parse(field_str, optimized_SLVL_mapping);
    }

    bool SetLidsPerPort(const char *field_str) {
       return CsvParser::Parse(field_str, lids_per_port);
    }

    bool SetPartEnfCap(const char *field_str) {
       return CsvParser::Parse(field_str, part_enf_cap);
    }

    bool SetInbEnfCap(const char *field_str) {
       return CsvParser::Parse(field_str, inb_enf_cap);
    }

    bool SetOutbEnfCap(const char *field_str) {
       return CsvParser::Parse(field_str, outb_enf_cap);
    }

    bool SetFilterRawInbCap(const char *field_str) {
       return CsvParser::Parse(field_str, filter_raw_inb_cap);
    }

    bool SetFilterRawOutbCap(const char *field_str) {
       return CsvParser::Parse(field_str, filter_raw_outb_cap);
    }

    bool SetENP0(const char *field_str) {
       return CsvParser::Parse(field_str, en_port0);
    }

    bool SetMCastFDBTop(const char *field_str) {
       return CsvParser::Parse(field_str, mcast_FDB_top);
    }
};

class LinkRecord {

public:

    u_int64_t           node_guid1;
    u_int8_t            port_num1;
    u_int64_t           node_guid2;
    u_int8_t            port_num2;

    LinkRecord()
        : node_guid1(0), port_num1(0), node_guid2(0), port_num2(0)
    {}

    static int Init(vector < ParseFieldInfo <class LinkRecord> > &parse_section_info);

    bool SetNodeGuid1(const char *field_str) {
        return CsvParser::Parse(field_str, node_guid1, 16);
    }

    bool SetPortNum1(const char *field_str) {
        return CsvParser::Parse(field_str, port_num1);
    }

    bool SetNodeGuid2(const char *field_str) {
        return CsvParser::Parse(field_str, node_guid2, 16);
    }

    bool SetPortNum2(const char *field_str) {
        return CsvParser::Parse(field_str, port_num2);
    }
};

class GeneralInfoSMPRecord {
public:
    u_int64_t           node_guid;
    string              fw_info_extended_major;
    string              fw_info_extended_minor;
    string              fw_info_extended_sub_minor;
    string              capability_mask_fields[NUM_CAPABILITY_FIELDS];

    GeneralInfoSMPRecord() : node_guid(0) {}

    static int Init(vector < ParseFieldInfo <class GeneralInfoSMPRecord> > &parse_section_info);

    bool SetNodeGUID(const char *field_str) {
        return CsvParser::Parse(field_str, node_guid, 16);
    }

    bool SetFWInfoExtendedMajor(const char *field_str) {
        return CsvParser::Parse(field_str, fw_info_extended_major);
    }

    bool SetFWInfoExtendedMinor(const char *field_str) {
        return CsvParser::Parse(field_str, fw_info_extended_minor);
    }

    bool SetFWInfoExtendedSubMinor(const char *field_str) {
        return CsvParser::Parse(field_str, fw_info_extended_sub_minor);
    }

    bool SetCapabilityMaskField0(const char *field_str) {
        return CsvParser::Parse(field_str, capability_mask_fields[0]);
    }

    bool SetCapabilityMaskField1(const char *field_str) {
        return CsvParser::Parse(field_str, capability_mask_fields[1]);
    }

    bool SetCapabilityMaskField2(const char *field_str) {
        return CsvParser::Parse(field_str, capability_mask_fields[2]);
    }

    bool SetCapabilityMaskField3(const char *field_str) {
        return CsvParser::Parse(field_str, capability_mask_fields[3]);
    }
};

class GeneralInfoGMPRecord {
public:
    u_int64_t           node_guid;
    string              hw_info_device_id;
    string              hw_info_device_hw_revision;
    u_int8_t            hw_info_technology;
    string              hw_info_up_time;
    string              fw_info_sub_minor;
    string              fw_info_minor;
    string              fw_info_major;
    string              fw_info_build_id;
    string              fw_info_year;
    string              fw_info_day;
    string              fw_info_month;
    string              fw_info_hour;
    string              fw_info_psid;
    string              fw_info_ini_file_version;
    string              fw_info_extended_major;
    string              fw_info_extended_minor;
    string              fw_info_extended_sub_minor;
    string              sw_info_sub_minor;
    string              sw_info_minor;
    string              sw_info_major;
    string              capability_mask_fields[NUM_CAPABILITY_FIELDS];

    GeneralInfoGMPRecord() : node_guid(0), hw_info_technology(0) {}

    static int Init(vector < ParseFieldInfo <class GeneralInfoGMPRecord> > &parse_section_info);

    bool SetNodeGUID(const char *field_str) {
        return CsvParser::Parse(field_str, node_guid, 16);
    }

    bool SetHWInfoDeviceID(const char *field_str) {
        return CsvParser::Parse(field_str,  hw_info_device_id);
    }

    bool SetHWInfoDeviceHWRevision(const char *field_str) {
        return CsvParser::Parse(field_str, hw_info_device_hw_revision);
    }

    bool SetHWInfoTechnology(const char *field_str) {
        return CsvParser::Parse(field_str, hw_info_technology);
    }

    bool SetHWInfoUpTime(const char *field_str) {
        return CsvParser::Parse(field_str, hw_info_up_time);
    }

    bool SetFWInfoSubMinor(const char *field_str) {
        return CsvParser::Parse(field_str, fw_info_sub_minor);
    }

    bool SetFWInfoMinor(const char *field_str) {
        return CsvParser::Parse(field_str, fw_info_minor);
    }

    bool SetFWInfoMajor(const char *field_str) {
        return CsvParser::Parse(field_str, fw_info_major);
    }

    bool SetFWInfoBuildID(const char *field_str) {
        return CsvParser::Parse(field_str, fw_info_build_id);
    }

    bool SetFWInfoYear(const char *field_str) {
        return CsvParser::Parse(field_str, fw_info_year);
    }

    bool SetFWInfoDay(const char *field_str) {
        return CsvParser::Parse(field_str, fw_info_day);
    }

    bool SetFWInfoMonth(const char *field_str) {
        return CsvParser::Parse(field_str, fw_info_month);
    }

    bool SetFWInfoHour(const char *field_str) {
        return CsvParser::Parse(field_str, fw_info_hour);
    }

    bool SetFWInfoPSID(const char *field_str) {
        return CsvParser::Parse(field_str, fw_info_psid);
    }

    bool SetFWInfoINIFileVersion(const char *field_str) {
        return CsvParser::Parse(field_str, fw_info_ini_file_version);
    }

    bool SetFWInfoExtendedMajor(const char *field_str) {
        return CsvParser::Parse(field_str, fw_info_extended_major);
    }

    bool SetFWInfoExtendedMinor(const char *field_str) {
        return CsvParser::Parse(field_str, fw_info_extended_minor);
    }

    bool SetFWInfoExtendedSubMinor(const char *field_str) {
        return CsvParser::Parse(field_str, fw_info_extended_sub_minor);
    }

    bool SetSWInfoSubMinor(const char *field_str) {
        return CsvParser::Parse(field_str, sw_info_sub_minor);
    }

    bool SetSWInfoMinor(const char *field_str) {
        return CsvParser::Parse(field_str, sw_info_minor);
    }

    bool SetSWInfoMajor(const char *field_str) {
        return CsvParser::Parse(field_str, sw_info_major);
    }

    bool SetCapabilityMaskField0(const char *field_str) {
        return CsvParser::Parse(field_str, capability_mask_fields[0]);
    }

    bool SetCapabilityMaskField1(const char *field_str) {
        return CsvParser::Parse(field_str, capability_mask_fields[1]);
    }

    bool SetCapabilityMaskField2(const char *field_str) {
        return CsvParser::Parse(field_str, capability_mask_fields[2]);
    }

    bool SetCapabilityMaskField3(const char *field_str) {
        return CsvParser::Parse(field_str, capability_mask_fields[3]);
    }
};

class ExtendedNodeInfoRecord
{
public:

    u_int64_t    node_guid;
    u_int8_t     sl2vl_cap;
    u_int8_t     sl2vl_act;
    u_int8_t     num_pcie;
    u_int8_t     num_oob;
    u_int8_t     AnycastLIDTop;
    u_int8_t     AnycastLidCap;
    u_int8_t     node_type_extended;
    u_int8_t     asic_max_planes;

    ExtendedNodeInfoRecord() : node_guid(0), sl2vl_cap(0), sl2vl_act(0),
        num_pcie(0), num_oob(0), AnycastLIDTop(0), AnycastLidCap(0), node_type_extended(0), asic_max_planes(0)
    {}

    static int Init(vector < ParseFieldInfo <class ExtendedNodeInfoRecord> > &parse_section_info);

    bool SetNodeGUID(const char *field_str) {
        return CsvParser::Parse(field_str, node_guid, 16);
    }

    bool SetSL2VLCap(const char *field_str) {
        return CsvParser::Parse(field_str, sl2vl_cap, 16);
    }

    bool SetSL2VLAct(const char *field_str) {
        return CsvParser::Parse(field_str, sl2vl_act, 16);
    }

    bool SetNumPCIe(const char *field_str) {
        return CsvParser::Parse(field_str, num_pcie, 10);
    }

    bool SetNumOOB(const char *field_str) {
        return CsvParser::Parse(field_str, num_oob, 10);
    }

    bool SetAnycastLIDTop(const char *field_str) {
        return CsvParser::Parse(field_str, AnycastLIDTop, 10);
    }

    bool SetAnycastLidCap(const char *field_str) {
        return CsvParser::Parse(field_str, AnycastLidCap, 10);
    }

    bool SetNodeTypeExtended(const char *field_str) {
        return CsvParser::Parse(field_str, node_type_extended, 10);
    }

    bool SetAsicMaxPlanes(const char *field_str) {
        return CsvParser::Parse(field_str, asic_max_planes, 10);
    }
};

class ExtendedPortInfoRecord {
public:
    u_int64_t           node_guid;
    u_int64_t           port_guid;
    u_int8_t            port_num;
    u_int8_t            state_change_enable;
    u_int8_t            sharp_an_en;
    u_int8_t            router_lid_en;
    u_int8_t            ame;
    u_int8_t            link_speed_supported;
    u_int8_t            unhealthy_reason;
    u_int8_t            link_speed_enabled;
    u_int8_t            link_speed_active;
    u_int16_t           active_rsfec_parity;
    u_int16_t           active_rsfec_data;
    u_int16_t           capability_mask;
    u_int8_t            fec_mode_active;
    u_int8_t            retrans_mode;
    u_int16_t           fdr10_fec_mode_supported;
    u_int16_t           fdr10_fec_mode_enabled;
    u_int16_t           fdr_fec_mode_supported;
    u_int16_t           fdr_fec_mode_enabled;
    u_int16_t           edr20_fec_mode_supported;
    u_int16_t           edr20_fec_mode_enabled;
    u_int16_t           edr_fec_mode_supported;
    u_int16_t           edr_fec_mode_enabled;
    u_int8_t            fdr10_retran_supported;
    u_int8_t            fdr10_retran_enabled;
    u_int8_t            fdr_retran_supported;
    u_int8_t            fdr_retran_enabled;
    u_int8_t            edr20_retran_supported;
    u_int8_t            edr20_retran_enabled;
    u_int8_t            edr_retran_supported;
    u_int8_t            edr_retran_enabled;
    u_int8_t            is_special_port;
    u_int8_t            special_port_type;
    u_int8_t            special_port_capability_mask;
    u_int8_t            is_fnm_port;
    u_int16_t           hdr_fec_mode_supported;
    u_int16_t           hdr_fec_mode_enabled;
    u_int16_t           ooosl_mask;
    u_int16_t           adaptive_timeout_sl_mask;
    u_int16_t           ndr_fec_mode_supported;
    u_int16_t           ndr_fec_mode_enabled;

    ExtendedPortInfoRecord()
        : node_guid(0), port_guid(0), port_num(0), state_change_enable(0), sharp_an_en(0),
          router_lid_en(0), ame(0), link_speed_supported(0),
          unhealthy_reason(0),link_speed_enabled(0),
          link_speed_active(0), active_rsfec_parity(0), active_rsfec_data(0),
          capability_mask(0), fec_mode_active(0), retrans_mode(0),
          fdr10_fec_mode_supported(0), fdr10_fec_mode_enabled(0),
          fdr_fec_mode_supported(0), fdr_fec_mode_enabled(0),
          edr20_fec_mode_supported(0), edr20_fec_mode_enabled(0),
          edr_fec_mode_supported(0), edr_fec_mode_enabled(0),
          fdr10_retran_supported(0), fdr10_retran_enabled(0),
          fdr_retran_supported(0), fdr_retran_enabled(0),
          edr20_retran_supported(0), edr20_retran_enabled(0),
          edr_retran_supported(0), edr_retran_enabled(0),
          is_special_port(0), special_port_type(0), special_port_capability_mask(0),
          is_fnm_port(0), hdr_fec_mode_supported(0), hdr_fec_mode_enabled(0), ooosl_mask(0),
          adaptive_timeout_sl_mask(0), ndr_fec_mode_supported(0),
          ndr_fec_mode_enabled(0)
    {}

    static int Init(vector < ParseFieldInfo <class ExtendedPortInfoRecord> > &parse_section_info);

    bool SetNodeGuid(const char *field_str) {
        return CsvParser::Parse(field_str, node_guid, 16);
    }

    bool SetPortGuid(const char *field_str) {
        return CsvParser::Parse(field_str, port_guid, 16);
    }

    bool SetPortNum(const char *field_str) {
        return CsvParser::Parse(field_str, port_num);
    }

    bool SetStateChangeEnable(const char *field_str) {
        return CsvParser::Parse(field_str, state_change_enable, 16);
    }

    bool SetSharpAnEn(const char *field_str) {
        return CsvParser::Parse(field_str, sharp_an_en, 16);
    }

    bool SetRouterLIDEn(const char *field_str) {
        return CsvParser::Parse(field_str, router_lid_en, 16);
    }

    bool SetAME(const char *field_str) {
        return CsvParser::Parse(field_str, ame, 16);
    }

    bool SetLinkSpeedSupported(const char *field_str) {
        return CsvParser::Parse(field_str, link_speed_supported, 16);
    }

    bool SetUnhealthyReason(const char *field_str) {
        return CsvParser::Parse(field_str, unhealthy_reason);
    }

    bool SetLinkSpeedEnabled(const char *field_str) {
        return CsvParser::Parse(field_str, link_speed_enabled, 16);
    }

    bool SetLinkSpeedActive(const char *field_str) {
        return CsvParser::Parse(field_str, link_speed_active, 16);
    }

    bool SetActiveRSFECParity(const char *field_str) {
        return CsvParser::Parse(field_str, active_rsfec_parity, 16);
    }

    bool SetActiveRSFECData(const char *field_str) {
        return CsvParser::Parse(field_str, active_rsfec_data, 16);
    }

    bool SetCapabilityMask(const char *field_str) {
        return CsvParser::Parse(field_str, capability_mask, 16);
    }

    bool SetFECModeActive(const char *field_str) {
        return CsvParser::Parse(field_str, fec_mode_active, 16);
    }

    bool SetRetransMode(const char *field_str) {
        return CsvParser::Parse(field_str, retrans_mode, 16);
    }

    bool SetFDR10FECModeSupported(const char *field_str) {
        return CsvParser::Parse(field_str, fdr10_fec_mode_supported, 16);
    }

    bool SetFDR10FECModeEnabled(const char *field_str) {
        return CsvParser::Parse(field_str, fdr10_fec_mode_enabled, 16);
    }

    bool SetFDRFECModeSupported(const char *field_str) {
        return CsvParser::Parse(field_str, fdr_fec_mode_supported, 16);
    }

    bool SetFDRFECModeEnabled(const char *field_str) {
        return CsvParser::Parse(field_str, fdr_fec_mode_enabled, 16);
    }

    bool SetEDR20FECModeSupported(const char *field_str) {
        return CsvParser::Parse(field_str, edr20_fec_mode_supported, 16);
    }

    bool SetEDR20FECModeEnabled(const char *field_str) {
        return CsvParser::Parse(field_str, edr20_fec_mode_enabled, 16);
    }

    bool SetEDRFECModeSupported(const char *field_str) {
        return CsvParser::Parse(field_str, edr_fec_mode_supported, 16);
    }

    bool SetEDRFECModeEnabled(const char *field_str) {
        return CsvParser::Parse(field_str, edr_fec_mode_enabled, 16);
    }

    bool SetFDR10RetranSupported(const char *field_str) {
        return CsvParser::Parse(field_str, fdr10_retran_supported, 16);
    }

    bool SetFDR10RetranEnabled(const char *field_str) {
        return CsvParser::Parse(field_str, fdr10_retran_enabled, 16);
    }

    bool SetFDRRetranSupported(const char *field_str) {
        return CsvParser::Parse(field_str, fdr_retran_supported, 16);
    }

    bool SetFDRRetranEnabled(const char *field_str) {
        return CsvParser::Parse(field_str, fdr_retran_enabled, 16);
    }

    bool SetEDR20RetranSupported(const char *field_str) {
        return CsvParser::Parse(field_str, edr20_retran_supported, 16);
    }

    bool SetEDR20RetranEnabled(const char *field_str) {
        return CsvParser::Parse(field_str, edr20_retran_enabled, 16);
    }

    bool SetEDRRetranSupported(const char *field_str) {
        return CsvParser::Parse(field_str, edr_retran_supported, 16);
    }

    bool SetEDRRetranEnabled(const char *field_str) {
        return CsvParser::Parse(field_str, edr_retran_enabled, 16);
    }

    bool SetIsSpecialPort(const char *field_str) {
        return CsvParser::Parse(field_str, is_special_port);
    }

    bool SetSpecialPortType(const char *field_str) {
        return CsvParser::ParseWithNA(field_str, special_port_type);
    }

    bool SetSpecialPortCapabilityMask(const char *field_str) {
        return CsvParser::Parse(field_str, special_port_capability_mask, 16);
    }

    bool SetFNMPort(const char *field_str) {
        return CsvParser::Parse(field_str, is_fnm_port);
    }

    bool SetHDRFECModeSupported(const char *field_str) {
        return CsvParser::Parse(field_str, hdr_fec_mode_supported, 16);
    }

    bool SetHDRFECModeEnabled(const char *field_str) {
        return CsvParser::Parse(field_str, hdr_fec_mode_enabled, 16);
    }

    bool SetOOOSLMask(const char *field_str) {
        return CsvParser::Parse(field_str, ooosl_mask, 16);
    }

    bool SetAdaptiveTimeoutSLMask(const char *field_str) {
        return CsvParser::Parse(field_str, adaptive_timeout_sl_mask, 16);
    }

    bool SetNDRFECModeSupported(const char *field_str) {
        return CsvParser::Parse(field_str, ndr_fec_mode_supported, 16);
    }

    bool SetNDRFECModeEnabled(const char *field_str) {
        return CsvParser::Parse(field_str, ndr_fec_mode_enabled, 16);
    }

};

class PortInfoExtendedRecord {
public:
    u_int64_t           node_guid;
    u_int64_t           port_guid;
    u_int8_t            port_num;
    u_int16_t           fec_mode_active;
    u_int16_t           fdr_fec_mode_supported;
    u_int16_t           fdr_fec_mode_enabled;
    u_int16_t           edr_fec_mode_supported;
    u_int16_t           edr_fec_mode_enabled;
    u_int16_t           hdr_fec_mode_supported;
    u_int16_t           hdr_fec_mode_enabled;
    u_int16_t           ndr_fec_mode_supported;
    u_int16_t           ndr_fec_mode_enabled;
    u_int32_t           capability_mask;

    PortInfoExtendedRecord()
        : node_guid(0), port_guid(0), port_num(0), fec_mode_active(0),
          fdr_fec_mode_supported(0), fdr_fec_mode_enabled(0),
          edr_fec_mode_supported(0), edr_fec_mode_enabled(0),
          hdr_fec_mode_supported(0), hdr_fec_mode_enabled(0),
          ndr_fec_mode_supported(0), ndr_fec_mode_enabled(0),
          capability_mask(0)
    {}

    static int Init(vector < ParseFieldInfo <class PortInfoExtendedRecord> > &parse_section_info);

    bool SetNodeGuid(const char *field_str) {
        return CsvParser::Parse(field_str, node_guid, 16);
    }

    bool SetPortGuid(const char *field_str) {
        return CsvParser::Parse(field_str, port_guid, 16);
    }

    bool SetPortNum(const char *field_str) {
        return CsvParser::Parse(field_str, port_num);
    }

    bool SetFECModeActive(const char *field_str) {
        return CsvParser::Parse(field_str, fec_mode_active, 16);
    }

    bool SetFDRFECModeSupported(const char *field_str) {
        return CsvParser::Parse(field_str, fdr_fec_mode_supported, 16);
    }

    bool SetFDRFECModeEnabled(const char *field_str) {
        return CsvParser::Parse(field_str, fdr_fec_mode_enabled, 16);
    }

    bool SetEDRFECModeSupported(const char *field_str) {
        return CsvParser::Parse(field_str, edr_fec_mode_supported, 16);
    }

    bool SetEDRFECModeEnabled(const char *field_str) {
        return CsvParser::Parse(field_str, edr_fec_mode_enabled, 16);
    }

    bool SetHDRFECModeSupported(const char *field_str) {
        return CsvParser::Parse(field_str, hdr_fec_mode_supported, 16);
    }

    bool SetHDRFECModeEnabled(const char *field_str) {
        return CsvParser::Parse(field_str, hdr_fec_mode_enabled, 16);
    }

    bool SetNDRFECModeSupported(const char *field_str) {
        return CsvParser::Parse(field_str, ndr_fec_mode_supported, 16);
    }

    bool SetNDRFECModeEnabled(const char *field_str) {
        return CsvParser::Parse(field_str, ndr_fec_mode_enabled, 16);
    }

    bool SetCapabilityMask(const char *field_str) {
        return CsvParser::Parse(field_str, capability_mask, 16);
    }
};

// AR Info
class ARInfoRecord {

public:
    u_int64_t          node_guid;
    u_int8_t           e;
    u_int8_t           is_arn_sup;
    u_int8_t           is_frn_sup;
    u_int8_t           is_fr_sup;
    u_int8_t           fr_enabled;
    u_int8_t           rn_xmit_enabled;
    u_int8_t           is_ar_trials_supported;
    u_int8_t           sub_grps_active;
    u_int8_t           group_table_copy_sup;
    u_int8_t           direction_num_sup;
    u_int8_t           is4_mode;
    u_int8_t           glb_groups;
    u_int8_t           by_sl_cap;
    u_int8_t           by_sl_en;
    u_int8_t           by_transp_cap;
    u_int8_t           dyn_cap_calc_sup;
    u_int16_t          group_cap;
    u_int16_t          group_top;
    u_int8_t           group_table_cap;
    u_int8_t           string_width_cap;
    u_int8_t           ar_version_cap;
    u_int8_t           rn_version_cap;
    u_int8_t           sub_grps_supported;
    u_int16_t          enable_by_sl_mask;
    u_int8_t           by_transport_disable;
    u_int32_t          ageing_time_value;
    u_int8_t           is_whbf_supported;
    u_int8_t           whbf_en;
    u_int8_t           is_hbf_supported;
    u_int8_t           by_sl_hbf_en;
    u_int16_t          enable_by_sl_mask_hbf;
    u_int8_t           whbf_granularity;
    u_int8_t           is_symmetric_hash_supported;
    u_int8_t           is_dceth_hash_supported;
    u_int8_t           is_bth_dqp_hash_supported;
    u_int8_t           is_pfrn_supported;
    u_int8_t           pfrn_enabled;

    ARInfoRecord()
        : node_guid(0), e(0), is_arn_sup(0), is_frn_sup(0), is_fr_sup(0), fr_enabled(0),
          rn_xmit_enabled(0), is_ar_trials_supported(0), sub_grps_active(0), group_table_copy_sup(0),
          direction_num_sup(0), is4_mode(0), glb_groups(0), by_sl_cap(0), by_sl_en(0),by_transp_cap(0),
          dyn_cap_calc_sup(0), group_cap(0),  group_top(0), group_table_cap(0), string_width_cap(0),
          ar_version_cap(0), rn_version_cap(0), sub_grps_supported(0), enable_by_sl_mask(0),
          by_transport_disable(0), ageing_time_value(0), is_whbf_supported(0), whbf_en(0),
          is_hbf_supported(0), by_sl_hbf_en(0), enable_by_sl_mask_hbf(0), whbf_granularity(0),
          is_symmetric_hash_supported(0), is_dceth_hash_supported(0), is_bth_dqp_hash_supported(0),
          is_pfrn_supported(0), pfrn_enabled(0)

    {}


    static int Init(vector < ParseFieldInfo <class ARInfoRecord> > &parse_section_info);

    bool SetNodeGuid(const char *field_str) {
        return CsvParser::Parse(field_str, node_guid, 16);
    }

    bool Set_e(const char *field_str) {
        return CsvParser::Parse(field_str, e, 10);
    }

    bool Set_is_arn_sup(const char *field_str) {
        return CsvParser::Parse(field_str, is_arn_sup, 10);
    }

    bool Set_is_frn_sup(const char *field_str) {
        return CsvParser::Parse(field_str, is_frn_sup, 10);
    }

    bool Set_is_fr_sup(const char *field_str) {
        return CsvParser::Parse(field_str, is_fr_sup, 10);
    }

    bool Set_fr_enabled(const char *field_str) {
        return CsvParser::Parse(field_str, fr_enabled, 10);
    }

    bool Set_rn_xmit_enabled(const char *field_str) {
        return CsvParser::Parse(field_str, rn_xmit_enabled, 10);
    }

    bool Set_is_ar_trials_supported(const char *field_str) {
        return CsvParser::Parse(field_str, is_ar_trials_supported, 10);
    }

    bool Set_sub_grps_active(const char *field_str) {
        return CsvParser::Parse(field_str, sub_grps_active, 10);
    }

    bool Set_group_table_copy_sup(const char *field_str) {
        return CsvParser::Parse(field_str, group_table_copy_sup, 10);
    }

    bool Set_direction_num_sup(const char *field_str) {
        return CsvParser::Parse(field_str, direction_num_sup, 10);
    }

    bool Set_is4_mode(const char *field_str) {
        return CsvParser::Parse(field_str, is4_mode, 10);
    }

    bool Set_glb_groups(const char *field_str) {
        return CsvParser::Parse(field_str, glb_groups, 10);
    }

    bool Set_by_sl_cap(const char *field_str) {
        return CsvParser::Parse(field_str, by_sl_cap, 10);
    }

    bool Set_by_sl_en(const char *field_str) {
        return CsvParser::Parse(field_str, by_sl_en, 10);
    }

    bool Set_by_transp_cap(const char *field_str) {
        return CsvParser::Parse(field_str, by_transp_cap, 10);
    }

    bool Set_dyn_cap_calc_sup(const char *field_str) {
        return CsvParser::Parse(field_str, dyn_cap_calc_sup, 10);
    }

    bool Set_group_cap(const char *field_str) {
        return CsvParser::Parse(field_str, group_cap, 10);
    }

    bool Set_group_top(const char *field_str) {
        return CsvParser::Parse(field_str, group_top, 10);
    }

    bool Set_group_table_cap(const char *field_str) {
        return CsvParser::Parse(field_str, group_table_cap, 10);
    }

    bool Set_string_width_cap(const char *field_str) {
        return CsvParser::Parse(field_str, string_width_cap, 10);
    }

    bool Set_ar_version_cap(const char *field_str) {
        return CsvParser::Parse(field_str, ar_version_cap, 10);
    }

    bool Set_rn_version_cap(const char *field_str) {
        return CsvParser::Parse(field_str, rn_version_cap, 10);
    }

    bool Set_sub_grps_supported(const char *field_str) {
        return CsvParser::Parse(field_str, sub_grps_supported, 10);
    }

    bool Set_enable_by_sl_mask(const char *field_str) {
        return CsvParser::Parse(field_str, enable_by_sl_mask, 10);
    }

    bool Set_by_transport_disable(const char *field_str) {
        return CsvParser::Parse(field_str, by_transport_disable, 10);
    }

    bool Set_ageing_time_value(const char *field_str) {
        return CsvParser::Parse(field_str, ageing_time_value, 10);
    }

    bool Set_is_whbf_supported(const char *field_str) {
        return CsvParser::Parse(field_str, is_whbf_supported, 10);
    }

    bool Set_whbf_en(const char *field_str) {
        return CsvParser::Parse(field_str, whbf_en, 10);
    }

    bool Set_is_hbf_supported(const char *field_str) {
        return CsvParser::Parse(field_str, is_hbf_supported, 10);
    }

    bool Set_by_sl_hbf_en(const char *field_str) {
        return CsvParser::Parse(field_str, by_sl_hbf_en, 10);
    }

    bool Set_enable_by_sl_mask_hbf(const char *field_str) {
        return CsvParser::Parse(field_str, enable_by_sl_mask_hbf, 10);
    }

    bool Set_whbf_granularity(const char *field_str) {
        return CsvParser::Parse(field_str, whbf_granularity, 10);
    }

    bool Set_is_symmetric_hash_supported(const char *field_str) {
        return CsvParser::Parse(field_str, is_symmetric_hash_supported, 10);
    }

    bool Set_is_dceth_hash_supported(const char *field_str) {
        return CsvParser::Parse(field_str, is_dceth_hash_supported, 10);
    }

    bool Set_is_bth_dqp_hash_supported(const char *field_str) {
        return CsvParser::Parse(field_str, is_bth_dqp_hash_supported, 10);
    }

    bool Set_is_pfrn_supported(const char *field_str) {
        return CsvParser::Parse(field_str, is_pfrn_supported, 10);
    }

    bool Set_pfrn_enabled(const char *field_str) {
        return CsvParser::Parse(field_str, pfrn_enabled, 10);
    }
};

class PortHierarchyInfoRecord
{
    public:
        u_int64_t m_node_guid;
        u_int64_t m_port_guid;
        u_int64_t m_template_guid;
        u_int8_t  m_port_num;

    public:
        int32_t  m_bus;
        int32_t  m_device;
        int32_t  m_function;
        int32_t  m_type;
        int32_t  m_slot_type;
        int32_t  m_slot_value;
        int32_t  m_asic;
        int32_t  m_cage;
        int32_t  m_port;
        int32_t  m_split;
        int32_t  m_ibport;
        int32_t  m_port_type;
        int32_t  m_asic_name;
        int32_t  m_is_cage_manager;
        int32_t  m_number_on_base_board;


        // APort
        int32_t  m_aport;
        int32_t  m_plane;
        int32_t  m_num_of_planes;

    public:
        PortHierarchyInfoRecord()
            : m_node_guid(0), m_port_guid(0), m_template_guid(0),
              m_port_num(0), m_bus(0), m_device(0), m_function(0),
              m_type(0), m_slot_type(0), m_slot_value(0),
              m_asic(0), m_cage(0), m_port(0), m_split(0), 
              m_ibport(0), m_port_type(0), m_asic_name(0),
              m_is_cage_manager(0), m_number_on_base_board(0),
              m_aport(0), m_plane(0), m_num_of_planes(0)

        {}

    public:
        static int Init(vector < ParseFieldInfo <class PortHierarchyInfoRecord> > &parse_section_info);

        bool SetNodeGUID(const char *field_str) {
            return CsvParser::Parse(field_str, m_node_guid, 16);
        }

        bool SetPortGUID(const char *field_str) {
            return CsvParser::Parse(field_str, m_port_guid, 16);
        }

        bool SetTemplateGUID(const char *field_str) {
            return CsvParser::Parse(field_str, m_template_guid, 16);
        }

        bool SetPortNum(const char *field_str) {
            return CsvParser::Parse(field_str, m_port_num, 10);
        }

        bool SetBus(const char *field_str) {
            return CsvParser::ParseWithNA(field_str, m_bus, -1, 10);
        }

        bool SetDevice(const char *field_str) {
            return CsvParser::ParseWithNA(field_str, m_device, -1, 10);
        }

        bool SetFunction(const char *field_str) {
            return CsvParser::ParseWithNA(field_str, m_function, -1, 10);
        }

        bool SetType(const char *field_str) {
            return CsvParser::ParseWithNA(field_str, m_type, -1, 10);
        }

        bool SetSlotType(const char *field_str) {
            return CsvParser::ParseWithNA(field_str, m_slot_type, -1, 10);
        }

        bool SetSlotValue(const char *field_str) {
            return CsvParser::ParseWithNA(field_str, m_slot_value, -1, 10);
        }

        bool SetIsCageManager(const char *field_str) {
            return CsvParser::ParseWithNA(field_str, m_is_cage_manager, -1, 10);
        }

        bool SetNumberOnBaseBoard(const char *field_str) {
            return CsvParser::ParseWithNA(field_str, m_number_on_base_board, -1, 10);
        }

        bool SetAPort(const char *field_str) {
            return CsvParser::ParseWithNA(field_str, m_aport, -1, 10);
        }

        bool SetPlane(const char *field_str) {
            return CsvParser::ParseWithNA(field_str, m_plane, -1, 10);
        }

        bool SetNumOfPlanes(const char *field_str) {
            return CsvParser::ParseWithNA(field_str, m_num_of_planes, -1, 10);
        }

        bool SetASIC(const char *field_str) {
            return CsvParser::ParseWithNA(field_str, m_asic, -1, 10);
        }

        bool SetCage(const char *field_str) {
            return CsvParser::ParseWithNA(field_str, m_cage, -1, 10);
        }

        bool SetPort(const char *field_str) {
            return CsvParser::ParseWithNA(field_str, m_port, -1, 10);
        }

        bool SetSplit(const char *field_str) {
            return CsvParser::ParseWithNA(field_str, m_split, -1, 10);
        }

        bool SetIBPort(const char *field_str) {
            return CsvParser::ParseWithNA(field_str, m_ibport, -1, 10);
        }

        bool SetPortType(const char *field_str) {
            return CsvParser::ParseWithNA(field_str, m_port_type, -1, 10);
        }

        bool SetAsicName(const char *field_str) {
            return CsvParser::ParseWithNA(field_str, m_asic_name, -1, 10);
        }
};

class PhysicalHierarchyInfoRecord
{
    public:
        u_int64_t m_node_guid;

    public:
        int32_t  m_campus_serial_num;
        int32_t  m_room_serial_num;
        int32_t  m_rack_serial_num;
        int32_t  m_system_type;
        int32_t  m_system_topu_num;
        int32_t  m_board_type;
        int32_t  m_board_slot_num;
        int32_t  m_device_serial_num;

    public:
        PhysicalHierarchyInfoRecord()
            :m_node_guid(0), m_campus_serial_num(0), m_room_serial_num(0),
             m_rack_serial_num(0), m_system_type(0), m_system_topu_num(0),
             m_board_type(0), m_board_slot_num(0), m_device_serial_num(0)
        {}


    public:
        static int Init(vector < ParseFieldInfo <class PhysicalHierarchyInfoRecord> > &parse_section_info);

        bool SetNodeGUID(const char *field_str) {
            return CsvParser::Parse(field_str, m_node_guid, 16);
        }

        bool SetCampusSerialNum(const char *field_str) {
            return CsvParser::ParseWithNA(field_str, m_campus_serial_num, -1, 10);
        }

        bool SetRoomSerialNum(const char *field_str) {
            return CsvParser::ParseWithNA(field_str, m_room_serial_num, -1, 10);
        }

        bool SetRackSerialNum(const char *field_str) {
            return CsvParser::ParseWithNA(field_str, m_rack_serial_num, -1, 10);
        }

        bool SetSystemType(const char *field_str) {
            return CsvParser::ParseWithNA(field_str, m_system_type, -1, 10);
        }

        bool SetSystemTopUNum(const char *field_str) {
            return CsvParser::ParseWithNA(field_str, m_system_topu_num, -1, 10);
        }

        bool SetBoardType(const char *field_str) {
            return CsvParser::ParseWithNA(field_str, m_board_type, -1, 10);
        }

        bool SetBoardSlotNum(const char *field_str) {
            return CsvParser::ParseWithNA(field_str, m_board_slot_num, -1, 10);
        }

        bool SetDeviceSerialNum(const char *field_str) {
            return CsvParser::ParseWithNA(field_str, m_device_serial_num, -1, 10);
        }
};

#endif   /* IBDIAG_FABRIC_H */
