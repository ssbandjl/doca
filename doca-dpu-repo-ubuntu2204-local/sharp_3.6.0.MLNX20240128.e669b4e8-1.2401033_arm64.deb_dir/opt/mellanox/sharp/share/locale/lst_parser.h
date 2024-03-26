/*
 * Copyright (c) 2004-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef LST_PARSER_H_
#define LST_PARSER_H_

#include <sys/types.h>
#include <functional>

#include "agg_types.h"
#include "csv_parser.h"
#include "fabric_db.h"
#include "fabric_graph.h"

#define EXIT_ON_ERROR     true
#define CONTINUE_ON_ERROR false

class FabricFileParser;
using VecFabricFileParserPtr = std::vector<std::unique_ptr<FabricFileParser>>;
using update_func_ptr = std::function<int()>;

class Fabric;   // the SHArP fabric containing Agg Nodes and trees
class FabricFileParser
{
   protected:
    string m_file_name;
    time_t m_st_mtime_;   // file creation time
    time_t m_last_mtime_;
    TopologyType m_topology_type;
    bool m_exit_on_error;
    bool m_should_reparse_file;

    update_func_ptr m_update_start;
    update_func_ptr m_update_end;
    update_func_ptr m_update_failed;

    CsvParser* m_csv_parser;
    FabricGraph* m_fabric_graph_;

   public:
    FabricFileParser(const string& file_name,
                     TopologyType topology_type,
                     bool exit_on_error,
                     update_func_ptr update_start,
                     update_func_ptr update_end,
                     update_func_ptr update_failed,
                     CsvParser* csv_parser,
                     FabricGraph* fabric_graph)
        : m_file_name(file_name),
          m_st_mtime_(0),
          m_last_mtime_(0),
          m_topology_type(topology_type),
          m_exit_on_error(exit_on_error),
          m_should_reparse_file(false),
          m_update_start(update_start),
          m_update_end(update_end),
          m_update_failed(update_failed),
          m_csv_parser(csv_parser),
          m_fabric_graph_(fabric_graph)
    {}
    virtual ~FabricFileParser() {}

    virtual int Load() = 0;

    bool CheckFileHasChanged();
    int OpenFile(ifstream& ifs);

    static int CheckFileExist(const string& file_name, uint8_t num_retries, uint8_t timeout);

    const string& GetFileName() { return m_file_name; }
    bool GetExitOnError() { return m_exit_on_error; }

    int LoadFabric();
    virtual int Init() { return 0; }
};

class SmdbFabricFileParser : public FabricFileParser
{
    SectionParser<SmRecord> sm_section_parser;
    SectionParser<NodeRecord> node_section_parser;
    SectionParser<PortRecord> port_section_parser;
    SectionParser<LinkRecord> link_section_parser;
    SectionParser<SwitchRecord> switch_section_parser;
    SectionParser<AnToAnRecord> an_to_an_section_parser;
    SectionParser<SmPortsRecord> sm_ports_section_parser;
    SectionParser<SmsRecord> sms_section_parser;

   public:
    SmdbFabricFileParser(const string& file_name,
                         TopologyType topology_type,
                         bool exit_on_error,
                         update_func_ptr update_start,
                         update_func_ptr update_end,
                         update_func_ptr update_failed,
                         CsvParser* csv_parser,
                         FabricGraph* fabric_graph)
        : FabricFileParser(file_name, topology_type, exit_on_error, update_start, update_end, update_failed, csv_parser, fabric_graph)
    {}

    virtual ~SmdbFabricFileParser() {}

    int Load();
    virtual int Init();
};

class VirtualizationFabricFileParser : public FabricFileParser
{
    SectionParser<VportRecord> vport_section_parser;
    SectionParser<VnodeRecord> vnode_section_parser;

   public:
    VirtualizationFabricFileParser(const string& file_name,
                                   TopologyType topology_type,
                                   bool exit_on_error,
                                   update_func_ptr update_start,
                                   update_func_ptr update_end,
                                   update_func_ptr update_failed,
                                   CsvParser* csv_parser,
                                   FabricGraph* fabric_graph)
        : FabricFileParser(file_name, topology_type, exit_on_error, update_start, update_end, update_failed, csv_parser, fabric_graph)
    {}

    virtual ~VirtualizationFabricFileParser() {}

    int Load();
    virtual int Init();
};

class LstFabricFileParser : public FabricFileParser
{
    string m_roots_guid_file;
    string m_hc_coordinates_file;

    int ParseSubnetLinks();
    int ParseHyperCubeCoordinatesFile(MapGuidToHCCoordinates& coordinates_map);

   public:
    LstFabricFileParser(const string& file_name,
                        TopologyType topology_type,
                        bool exit_on_error,
                        update_func_ptr update_start,
                        update_func_ptr update_end,
                        update_func_ptr update_failed,
                        CsvParser* csv_parser,
                        FabricGraph* fabric_graph,
                        const string& roots_guid_file,
                        const string& hc_coordinates_file)
        : FabricFileParser(file_name, topology_type, exit_on_error, update_start, update_end, update_failed, csv_parser, fabric_graph),
          m_roots_guid_file(roots_guid_file),
          m_hc_coordinates_file(hc_coordinates_file)
    {}
    virtual ~LstFabricFileParser() {}

    int Load();
};

class FilesBasedFabricDB : public FabricDb
{
    FabricGraph m_fabric_graph_;
    pthread_t m_parser_thread_;

    VecFabricFileParserPtr m_fabric_file_parser_vec;

   public:
    // Constructor
    FilesBasedFabricDB(CommandManager* command_manager_ptr) : FabricDb(), m_fabric_graph_(command_manager_ptr), m_parser_thread_(0) {}

    // destructor
    virtual ~FilesBasedFabricDB();

    FabricGraph& GetFabricGraph() { return m_fabric_graph_; }

    int Start();

    // return MAX_NUM_HOPS if no path found
    virtual uint8_t GetNumHops(uint64_t from_sw_guid, uint64_t to_sw_guid) override;

    CsvParser m_csv_parser;

   private:
    static void* MainLoop(void* p_parser);
    int CreateFileParsers();
};

#endif   // LST_PARSER_H_
