/*
 * Copyright (c) 2016-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef _FD_EVENT_LISTENER_H
#define _FD_EVENT_LISTENER_H

#include <poll.h>
#include <pthread.h>
#include <list>
#include <map>

#include "sharp_common.h"

typedef std::list<class FdEvent*> ListEventPtr;

typedef void(handle_cb)(const void* delegate, void* context);

class FdEvent
{
   public:
    FdEvent(handle_cb* cb, const void* delegate, void* context)
        : m_fd_(-1), m_event_delegate_(delegate), m_event_context_(context), m_event_h_(cb)
    {}

    virtual ~FdEvent() {}

    virtual int Init() { return 0; }

    virtual void Handle() const
    {
        if (m_event_h_)
            m_event_h_(m_event_delegate_, m_event_context_);
    }

    int GetFd() const { return m_fd_; }

    void SetFd(int fd) { m_fd_ = fd; }

   private:
    int m_fd_;

    const void* m_event_delegate_;
    void* m_event_context_;
    handle_cb* m_event_h_;
};

class TimerEvent : public FdEvent
{
   public:
    TimerEvent(handle_cb* cb, const void* delegate, void* context, int timeout /* secs */)
        : FdEvent(cb, delegate, context), m_timeout_(timeout), m_timer_(NULL)
    {}

    ~TimerEvent();

    int Init();
    void Handle() const;

    void Start();
    void Stop();

   private:
    int m_timeout_;

    sharp_timer_ctx* m_timer_;
};

class FdEventListener
{
   public:
    explicit FdEventListener() : m_thread_(), m_stop_thread_(true) {}

    ~FdEventListener();

    int RegisterFdEvent(FdEvent* event);
    void UnregisterFdEvents();

    int Start();

    void StopListener();

   private:
    pthread_t m_thread_;
    bool m_stop_thread_;

    ListEventPtr m_events_;

    int StartListener();

    static void* Listener(void* context);
};

#endif /* _FD_EVENT_LISTENER_H */
