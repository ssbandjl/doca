/*
 * Copyright (c) 2004-2021 Mellanox Technologies LTD. All rights reserved.
 * Copyright (c) 2021-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef PROGRESS_BAR_H
#define PROGRESS_BAR_H

#include <map>
#include <stdint.h>
#include <time.h>
#include <infiniband/ibdm/Fabric.h>

class ProgressBar
{
    protected:
        typedef struct entry {
            uint64_t m_total;
            uint64_t m_complete;

            entry() : m_total(0), m_complete(0) {}
        } entry_t;

        typedef std::map<const IBPort*, uint64_t> ports_stat_t;
        typedef std::map<const IBNode*, uint64_t> nodes_stat_t;

        entry_t         m_sw;
        entry_t         m_ca;

        entry_t         m_sw_ports;
        entry_t         m_ca_ports;

        entry_t         m_requests;

        ports_stat_t    m_ports_stat;
        nodes_stat_t    m_nodes_stat;

        struct timespec m_last_update;

    public:
        ProgressBar() {
            clock_gettime(CLOCK_REALTIME, &m_last_update);
        }

        virtual ~ProgressBar() {}

    public:
        virtual void output() = 0;

    private:
        void update(bool force=false)
        {
            struct timespec now;
            clock_gettime(CLOCK_REALTIME, &now);

            if (now.tv_sec - m_last_update.tv_sec > 1 || force) {
                output();
                m_last_update = now;
            }
        }

    public:
        void push(const IBNode *node) {
            nodes_stat_t::iterator x = m_nodes_stat.find(node);

            if (x == m_nodes_stat.end()) {
                m_nodes_stat[node] = 1;

                if (node->type == IB_SW_NODE)
                    m_sw.m_total++;
                else
                    m_ca.m_total++;
            } else {
                if (!x->second) {
                    if (node->type == IB_SW_NODE)
                        m_sw.m_complete--;
                    else
                        m_ca.m_complete--;
                }

                x->second++;
            }

            m_requests.m_total++;
            update();
        }

        void push(const IBPort *port) {
            ports_stat_t::iterator x = m_ports_stat.find(port);

            if (x == m_ports_stat.end()) {
                m_ports_stat[port] = 1;

                if (port->p_node->type == IB_SW_NODE)
                    m_sw_ports.m_total++;
                else
                    m_ca_ports.m_total++;

                push(port->p_node);
            } else {
                if (!x->second) {
                    push(port->p_node);

                    if (port->p_node->type == IB_SW_NODE)
                        m_sw_ports.m_complete--;
                    else
                        m_ca_ports.m_complete--;
                } else {
                    m_requests.m_total++;
                    update();
                }

                x->second++;
            }
        }

    public:
        void complete(const IBNode *node) {
            nodes_stat_t::iterator x = m_nodes_stat.find(node);

            if (x == m_nodes_stat.end() || !x->second)
                return;

            x->second--;

            if (!x->second) {
                if (node->type == IB_SW_NODE)
                    m_sw.m_complete++;
                else
                    m_ca.m_complete++;
            }

            m_requests.m_complete++;
            update();
        }

        void complete(const IBPort *port) {
            ports_stat_t::iterator x = m_ports_stat.find(port);

            if (x == m_ports_stat.end() || !x->second)
                return;

            x->second--;

            if (!x->second) {
                complete(port->p_node);

                if (port->p_node->type == IB_SW_NODE)
                    m_sw_ports.m_complete++;
                else
                    m_ca_ports.m_complete++;
            } else {
                m_requests.m_complete++;
                update();
            }
        }

    public:
        template<typename T>
            static T *complete(void *p_progress_bar, void *p_obj) {
               T *p_t_obj = (T *)p_obj;
               if (p_progress_bar && p_t_obj)
                   ((ProgressBar *)p_progress_bar)->complete(p_t_obj);

               return p_t_obj;
            }
};

#endif          /* PROGRESS_BAR_H */
