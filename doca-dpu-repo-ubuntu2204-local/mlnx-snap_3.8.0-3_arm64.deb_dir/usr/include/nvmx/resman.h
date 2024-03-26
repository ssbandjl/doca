#ifndef RESMAN_H_
#define RESMAN_H_

/*
Resource manager

This class is a basic manager for resource of any kind.
Main purpose - fair distribution of limited resources
between threads basing on thread's demand.

The use concept:
    - A management thread creates a resman object and fills it
      with resource entries (like 1000 chunks of memory 1MB each)
    - A worker thread provides its demand, like Thread A wants
      100 entries, Thread B wants 15 entries etc.
    - Worker threads adjust their demands depending on load
    - Worker threads get entries from resman. Normally resman
      doesn't allow to get more than demanded
    - When all entries are given to threads and one or many
      threads get extra demand - resman does re-ballancing
      in a fair manner.
    - Normally each thread can get up to
       (thread_demand / total_demand) * num_total_entries
      Example:
        Total entries 50
        Thread A demand 100
        Thread B demand 15
        Total demand 115
        Result:
        Thread A gets (100 / 115) * 50 = 43
        Thread B gets (15 / 115) * 50 = 6
    - Some measures were introduced to avoid division losses
    - 100% Balance is rarely achieved since resman can't just
      take back an entry, it may be in use by the worker. If
      this is the case, resman signals the thread to return entry
 */

#include <stdbool.h>
#include "mlnx_atomic.h"
#include "lock_free_stack.h"

typedef struct {
    lfs_entry_t lfs_entry;
} resman_entry_t;

typedef struct resman_thread_ctx {
    size_t demand;              // Desired number of entries for this thread
                                // multiplied by resman->len
    lfs_stack_t stack;          // Thread local pool
} resman_thread_ctx_t;

typedef struct resman {
    size_t thread_count;        // The size for `levels` and `threads` arrays
    _Atomic size_t total_demand;// Sum of demands in entries of all the threads
    _Atomic int *levels;        // Represents amount of entries each thread
                                // received via resman_get
    resman_thread_ctx_t *threads;//Keeps thread contexts
    lfs_stack_t stack;          // Global pool
    size_t len;                 // Actual number of entries managed by resman
} resman_t;

//============================
// Management thread interface
//============================

// Initializes resman
int resman_init(resman_t *resman, size_t thread_count);

// Destroys resman
void resman_clear(resman_t *resman);

// Add entry to pool
int resman_add(resman_t *resman, resman_entry_t *entry);

//============================
// Worker thread interface
//============================

// Get entry from resman
// Returns pointer to entry if resman has available items and
// thread's quota is not exceeded
resman_entry_t *resman_get(resman_t *resman, size_t thread_id);

// Return entry to resman
bool resman_put(resman_t *resman, size_t thread_id, resman_entry_t *entry);

// Thread calls resman_inc_demand to increase it's demand by one entry
void resman_inc_demand(resman_t *resman, size_t thread_id);
// Thread calls resman_dec_demand to decrease  it's demand by one entry
void resman_dec_demand(resman_t *resman, size_t thread_id);

// Checks whether we got more than should
// Thread calls this function periodically in order to check whether
// it's quota has been changed and one or several entries should be
// returned to resman
bool resman_quota_exceed(resman_t *resman, size_t thread_id);

// Checks whether there is at least one available entry in global pool
// If positive, most likely resman_get will succeed
static inline bool resman_has_entry(resman_t *resman)
{
    return resman->stack.first != NULL;
}

#endif /* RESMAN_H_ */
