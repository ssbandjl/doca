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

#ifndef FABRIC_GRAPH_UPDATE_H_
#define FABRIC_GRAPH_UPDATE_H_

#include "agg_types.h"
#include "port_data.h"

struct FabricGraphPortDataUpdate;
class Fabric;

typedef std::list<FabricGraphPortDataUpdate> ListFabricGraphPortDataUpdate;

enum FabricGraphUpdateType
{
    FABRIC_GRAPH_UPDATE_CLEAN_ALL_REQ = 1,
};

static inline const char* FabricGraphUpdateType2Char(const FabricGraphUpdateType update_type)
{
    switch (update_type) {
        case FABRIC_GRAPH_UPDATE_CLEAN_ALL_REQ:
            return ("CLEAN ALL required");
        default:
            return ("invalid");
    }
};

struct FabricGraphPortDataUpdate
{
    port_key_t m_port_key;
    FabricGraphUpdateType m_update_type;
    string m_msg;   // Used to pass 'reason' for port update INACTIVE

    FabricGraphPortDataUpdate(port_key_t port_key, FabricGraphUpdateType update_type, string msg = "")
        : m_port_key(port_key), m_update_type(update_type), m_msg(msg)
    {}
};

class FabricGraphUpdateList
{
    ListFabricGraphPortDataUpdate m_ports_update_;
    pthread_mutex_t m_list_lock_;

   public:
    explicit FabricGraphUpdateList() { pthread_mutex_init(&m_list_lock_, NULL); }

    ~FabricGraphUpdateList(){};

    void AddUpdate(FabricGraphPortDataUpdate& port_update);
    void GetUpdates(ListFabricGraphPortDataUpdate& port_updates);
};

#endif   // FABRIC_GRAPH_UPDATE_H_
