/*
 * Copyright (c) 2012-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * This software is available to you under a choice of one of two
 * licenses.  You may choose to be licensed under the terms of the GNU
 * General Public License (GPL) Version 2, available from the file
 * COPYING in the main directory of this source tree, or the
 * OpenIB.org BSD license below:
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

#ifndef EVENT_MANAGER_H_
#define EVENT_MANAGER_H_

#include "smx/smx_api.h"

class CommandManager;

enum SharpEventSeverityEnum
{
    SHARP_EVENT_SEVERITY_INFO = 0,
    SHARP_EVENT_SEVERITY_WARNING,
    SHARP_EVENT_SEVERITY_ERROR,
};

static inline const char* SharpEventSeverityToChar(const SharpEventSeverityEnum event_severity)
{
    switch (event_severity) {
        case SHARP_EVENT_SEVERITY_INFO:
            return ("Info");
        case SHARP_EVENT_SEVERITY_WARNING:
            return ("Warning");
        case SHARP_EVENT_SEVERITY_ERROR:
            return ("Error");
        default:
            return ("UNKNOWN");
    }
};

struct SharpEvent
{
    sharp_event_type event_type;
    sharp_event* event;
    sharp_timestamp ts;

    SharpEvent();
    SharpEvent(sharp_event_type t, uint32_t num_of_str_entries);

    bool operator>(const sharp_timestamp& ts);
    bool operator<(const sharp_timestamp& ts);
    bool operator<=(const sharp_timestamp& ts);
    bool operator==(const sharp_timestamp& ts);

    ~SharpEvent();
};

class EventManager
{
    SharpEvent** m_cyclic_buffer_;   // buffer
    uint32_t m_size_;                // size of the buffer
    uint32_t m_tail_;                // index of next element insertion
    uint32_t m_head_;                // index of last element inserted
    pthread_mutex_t m_lock_;         // lock
    bool m_full_;                    // indication that the buffer is full

    CommandManager* m_command_manager_;

    int GetIndexLowerBound(const sharp_timestamp& ts);

   public:
    EventManager();

    int Init(uint32_t size);

    ~EventManager();

    // Reset the cyclic buffer, head == tail
    void Clear();

    // Add continues to add data if the buffer is full
    // Old data is overwritten
    void Add(SharpEvent* event);

    // Returns true if the buffer is empty
    bool IsEmpty();

    // Returns true if the buffer is full
    bool IsFull();

    // Returns the current number of elements in the buffer
    uint32_t GetSize();

    // Returns all events starting at ts timestamp, if ts == 0, returns all events
    uint32_t GetEventsByTimeStamp(const sharp_timestamp& ts, sharp_event_list*& events);

    // Frees all allocated memory for array of sharp_events
    void FreeSharpEventsSt(sharp_event_list* events);

    // Sets command manager object
    void SetCommandManager(CommandManager* command_manager);

    // Handles GetEvents SMX request. Sends matching events over SMX using CommandManager
    void HandleGetEventsByTimeStamp(const sharp_timestamp& ts, const smx_ep* ep, uint64_t tid);

    /* Handlers per specific event type*/
    void AddAmReadyEvent();
    void AddAmPendingModeEvent();
    void AddAggNodeDiscoveryFailedEvent(const char* node, const char* state);
    void AddJobStartedEvent(uint64_t external_job_id, const char* reservation_key, uint32_t sharp_job_id);
    void AddJobEndedEvent(uint64_t external_job_id, const char* reservation_key, uint32_t sharp_job_id);
    void AddJobStartFailedEvent(uint64_t external_job_id, const char* reservation_key, uint32_t sharp_job_id, const char* reason);
    void AddJobErrorEvent(uint64_t external_job_id, const char* reservation_key, uint32_t sharp_job_id, const char* err_msg);
    void AddReservationCreatedEvent(const char* reservation_key);
    void AddReservationUpdatedEvent(const char* reservation_key);
    void AddReservationRemovedEvent(const char* reservation_key);
    void AddTrapQPErrorEvent(uint16_t lid);
    void AddTrapInvalidRequestEvent(uint16_t lid);
    void AddTrapSharpErrorEvent(uint16_t lid);
    void AddTrapQPAllocTimeoutEvent(uint16_t lid);
    void AddTrapAMKeyViolationTriggeregByAMEvent(uint16_t lid);
    void AddTrapAMKeyViolationEvent(uint16_t lid);
    void AddTrapUnsupportedEvent(uint16_t lid);
    void AddAggNodeActiveEvent(const char* node, const char* switch_string);
    void AddAggNodeInactiveEvent(const char* node, const char* switch_string, const char* reason);
};

// Recognize the global variables
extern EventManager g_event_manager;

#endif   // EVENT_MANAGER_H_
