/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <sys/types.h>
#include "fabric_graph.h"

class Fabric;   // the SHArP fabric containing Agg Nodes and trees

struct AMGeneralInfoRecord
{
    string version;
    uint32_t retry_num;

    static int Init(vector<ParseFieldInfo<struct AMGeneralInfoRecord>>& parse_section_info);

    bool SetVersion(const char* field_str) { return CsvParser::Parse(field_str, version, 0); }

    bool SetRetryNum(const char* field_str) { return CsvParser::Parse(field_str, retry_num, 0); }

    AMGeneralInfoRecord()
    {
        version = VERSION;
        retry_num = 0;
    }
};

struct FileCRCRecord
{
    string filename;
    uint32_t crc;

    static int Init(vector<ParseFieldInfo<struct FileCRCRecord>>& parse_section_info);

    bool SetFileName(const char* field_str) { return CsvParser::Parse(field_str, filename, 0); }

    bool SetCRC(const char* field_str) { return CsvParser::Parse(field_str, crc, 0); }
};

struct AggPathRecord
{
    uint64_t guid1;
    uint32_t port1;
    uint64_t guid2;
    uint32_t port2;
    int action;

    static int Init(vector<ParseFieldInfo<struct AggPathRecord>>& parse_section_info);

    bool SetGuid1(const char* field_str) { return CsvParser::Parse(field_str, guid1, 0); }

    bool SetPort1(const char* field_str) { return CsvParser::Parse(field_str, port1, 0); }

    bool SetGuid2(const char* field_str) { return CsvParser::Parse(field_str, guid2, 0); }

    bool SetPort2(const char* field_str) { return CsvParser::Parse(field_str, port2, 0); }

    bool SetAction(const char* field_str) { return CsvParser::Parse(field_str, action, 0); }
};

struct AggNodeInfoRecord
{
    uint64_t node_guid;
    uint8_t max_control_path_version_supported;
    uint8_t active_control_path_version;
    uint16_t data_path_version_supported;
    uint16_t active_data_path_version;
    int multiple_sver_active_supported;
    uint32_t num_of_jobs;
    uint8_t radix;
    uint8_t max_radix;
    uint32_t osts;
    uint32_t buffers;
    uint32_t groups;
    uint32_t qps;
    uint32_t max_sat_qps;
    uint32_t max_llt_qps;
    uint32_t max_sat_qps_per_port;
    uint32_t max_llt_qps_per_port;
    uint32_t max_user_data_per_ost;
    uint8_t endianness;
    uint8_t reproducibility_disable_supported;
    int enable_reproducibility;
    int streaming_aggregation_supported;
    int configure_port_credit_supported;
    uint8_t num_semaphores;
    uint8_t num_active_semaphores;
    int semaphores_per_port;
    uint8_t reproducibility_per_job_supported;
    int enable_reproducibility_per_job;
    uint8_t am_key_supported;
    uint8_t job_key_supported;
    uint8_t tree_job_binding_supported;
    uint8_t tree_job_default_binding;
    uint16_t tree_table_size;
    int fp19_supported;
    int bfloat19_supported;
    int extended_data_types_supported;
    int mtu;

    static int Init(vector<ParseFieldInfo<struct AggNodeInfoRecord>>& parse_section_info);

    bool SetNodeGuid(const char* field_str) { return CsvParser::Parse(field_str, node_guid, 0); }

    bool SetMaxControlPathVersion(const char* field_str) { return CsvParser::Parse(field_str, max_control_path_version_supported, 0); }

    bool SetActiveControlPathVersion(const char* field_str) { return CsvParser::Parse(field_str, active_control_path_version, 0); }

    bool SetDataPathVersionSupported(const char* field_str) { return CsvParser::Parse(field_str, data_path_version_supported, 0); }

    bool SetActiveDataPathVersion(const char* field_str) { return CsvParser::Parse(field_str, active_data_path_version, 0); }

    bool SetMultipleSverActiveSupported(const char* field_str) { return CsvParser::Parse(field_str, multiple_sver_active_supported, 0); }

    bool SetNumOfJobs(const char* field_str) { return CsvParser::Parse(field_str, num_of_jobs, 0); }

    bool SetRadix(const char* field_str) { return CsvParser::Parse(field_str, radix, 0); }

    bool SetMaxRadix(const char* field_str) { return CsvParser::Parse(field_str, max_radix, 0); }

    bool SetOsts(const char* field_str) { return CsvParser::Parse(field_str, osts, 0); }

    bool SetBuffers(const char* field_str) { return CsvParser::Parse(field_str, buffers, 0); }

    bool SetGroups(const char* field_str) { return CsvParser::Parse(field_str, groups, 0); }

    bool SetQps(const char* field_str) { return CsvParser::Parse(field_str, qps, 0); }

    bool SetMaxSatQps(const char* field_str) { return CsvParser::Parse(field_str, max_sat_qps, 0); }

    bool SetMaxLltQps(const char* field_str) { return CsvParser::Parse(field_str, max_llt_qps, 0); }

    bool SetMaxSatQpsPerPort(const char* field_str) { return CsvParser::Parse(field_str, max_sat_qps_per_port, 0); }

    bool SetMaxLltQpsPerPort(const char* field_str) { return CsvParser::Parse(field_str, max_llt_qps_per_port, 0); }

    bool SetMaxUserDataPerOst(const char* field_str) { return CsvParser::Parse(field_str, max_user_data_per_ost, 0); }

    bool SetEndianess(const char* field_str) { return CsvParser::Parse(field_str, endianness, 0); }

    bool SetReproducobilityDisableSupported(const char* field_str)
    {
        return CsvParser::Parse(field_str, reproducibility_disable_supported, 0);
    }

    bool SetEnableReproducibility(const char* field_str) { return CsvParser::Parse(field_str, enable_reproducibility, 0); }

    bool SetStreamingAggregationSupported(const char* field_str) { return CsvParser::Parse(field_str, streaming_aggregation_supported, 0); }

    bool SetConfigurePortCreditSupported(const char* field_str) { return CsvParser::Parse(field_str, configure_port_credit_supported, 0); }

    bool SetNumSemaphores(const char* field_str) { return CsvParser::Parse(field_str, num_semaphores, 0); }

    bool SetNumActiveSemaphores(const char* field_str) { return CsvParser::Parse(field_str, num_active_semaphores, 0); }

    bool SetSemaphoresPerPort(const char* field_str) { return CsvParser::Parse(field_str, semaphores_per_port, 0); }

    bool SetReproducibilityPerJobSupported(const char* field_str)
    {
        return CsvParser::Parse(field_str, reproducibility_per_job_supported, 0);
    }

    bool SetEnableReproducibilityPerJob(const char* field_str) { return CsvParser::Parse(field_str, enable_reproducibility_per_job, 0); }

    bool SetAmKeySupported(const char* field_str) { return CsvParser::Parse(field_str, am_key_supported, 0); }

    bool SetJobKeySupported(const char* field_str) { return CsvParser::Parse(field_str, job_key_supported, 0); }

    bool SetTreeToJobBindingSupported(const char* field_str) { return CsvParser::Parse(field_str, tree_job_binding_supported, 0); }

    bool SetTreeToJobDefaultBinding(const char* field_str) { return CsvParser::Parse(field_str, tree_job_default_binding, 0); }

    bool SetTreeTableSize(const char* field_str) { return CsvParser::Parse(field_str, tree_table_size, 0); }

    bool SetFP19Supported(const char* field_str) { return CsvParser::Parse(field_str, fp19_supported, 0); }

    bool SetBFloat19Supported(const char* field_str) { return CsvParser::Parse(field_str, bfloat19_supported, 0); }

    bool SetExtendedDataTypesSupported(const char* field_str) { return CsvParser::Parse(field_str, extended_data_types_supported, 0); }

    bool SetMtu(const char* field_str) { return CsvParser::Parse(field_str, mtu, 0); }
};

struct AggNodeRecord
{
    uint64_t node_guid;
    string name;
    string state;
    string timestamp;

    AggNodeRecord() : node_guid(0) {}

    static int Init(vector<ParseFieldInfo<struct AggNodeRecord>>& parse_section_info);

    bool SetNodeGuid(const char* field_str) { return CsvParser::Parse(field_str, node_guid, 0); }

    bool SetName(const char* field_str) { return CsvParser::Parse(field_str, name); }

    bool SetState(const char* field_str) { return CsvParser::Parse(field_str, state); }

    bool SetTimestamp(const char* field_str) { return CsvParser::Parse(field_str, timestamp); }
};

class FabricDumpFileParser
{
   protected:
    string m_file_name;
    CsvParser* m_csv_parser;

   public:
    SectionParser<AggNodeRecord> agg_nodes_section_parser;
    SectionParser<AggNodeInfoRecord> agg_node_info_section_parser;
    SectionParser<AggPathRecord> agg_path_section_parser;
    SectionParser<FileCRCRecord> file_crc_parser;
    SectionParser<AMGeneralInfoRecord> general_info_parser;

    FabricDumpFileParser(const string& file_name) : m_file_name(file_name) { m_csv_parser = NULL; }
    ~FabricDumpFileParser() {}

    int Load();

    int CheckFileExist(uint8_t num_retries, uint8_t timeout);

    const string& GetFileName() { return m_file_name; }

    int LoadFabricDump();
    int Init();
};
