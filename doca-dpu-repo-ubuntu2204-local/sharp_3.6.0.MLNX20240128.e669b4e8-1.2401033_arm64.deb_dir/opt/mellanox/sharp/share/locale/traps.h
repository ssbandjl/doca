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

#ifndef AGG_TRAPS_H_
#define AGG_TRAPS_H_

#include <pthread.h>
#include <queue>

#include "agg_ib_types.h"
#include "agg_types.h"
#include "fabric_update.h"
#include "ibis.h"

class CommandManager;
class TreeNode;

typedef std::queue<class TrapHandler*> QueueTrapHandlerPtr;

class TrapHandler
{
   public:
    virtual void Handle(CommandManager& command_manager) = 0;

    virtual ~TrapHandler() {}
};

class QpErrorTrapHandler : public TrapHandler
{
    TrapQpError m_qp_error_;

   public:
    QpErrorTrapHandler(TrapQpError& qp_error) : m_qp_error_(qp_error) {}

    virtual void Handle(CommandManager& command_manager);
    virtual ~QpErrorTrapHandler() {}

    static TreeNode* GetTreeNode(lid_t an_port_lid, sharp_trees_t tree_id);
};

class InvalidReqTrapHandler : public TrapHandler
{
    sharp_job_id_t m_sharp_job_id_;

   public:
    InvalidReqTrapHandler(sharp_job_id_t sharp_job_id) : m_sharp_job_id_(sharp_job_id) {}

    virtual void Handle(CommandManager& command_manager);
    virtual ~InvalidReqTrapHandler() {}
};

class SharpErrorTrapHandler : public TrapHandler
{
    TrapSharpError m_sharp_error_;

   public:
    SharpErrorTrapHandler(TrapSharpError sharp_error) : m_sharp_error_(sharp_error) {}

    virtual void Handle(CommandManager& command_manager);
    virtual ~SharpErrorTrapHandler() {}
};

class TrapsQueue
{
    CommandManager& m_command_manager;

    QueueTrapHandlerPtr m_handlers_;
    pthread_mutex_t m_queue_lock_;

    bool m_agg_node_lids_array[FABRIC_MAX_VALID_LID];
    pthread_mutex_t m_agg_node_lids_array_lock_;

   public:
    TrapsQueue(CommandManager& command_manager) : m_command_manager(command_manager)
    {
        pthread_mutex_init(&m_queue_lock_, NULL), pthread_mutex_init(&m_agg_node_lids_array_lock_, NULL),
            memset(m_agg_node_lids_array, 0, sizeof(bool) * FABRIC_MAX_VALID_LID);
    }

    ~TrapsQueue();

    void Register();
    void AddTrap(SharpTrapNumberEnum trap_number,
                 void* p_data,
                 ib_address_t* p_ib_address,
                 MAD_AggregationManagement* p_am_mad,
                 Notice* p_notice);
    void HandleTraps();
    bool IsAggNode(uint16_t lid);
    void HandleAggNodeUpdateArray(const PortInfo& port_info, bool val);

   private:
    void AddTrap(TrapHandler* p_trap_handler);
};

#endif   // AGG_TRAPS_H_
