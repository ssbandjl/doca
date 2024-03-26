#ifndef SNAP_SYS_QUEUE_EXT_H_
#define SNAP_SYS_QUEUE_EXT_H_
#include <sys/queue.h>

/*
 * Extend queue.h functions with macros not included in sys/queue.h
 */
#ifdef HAVE_SPDK
#include <spdk/queue_extras.h>
#else
#ifndef TAILQ_FOREACH_SAFE
#define TAILQ_FOREACH_SAFE(var, head, field, next)                       \
        for ((var) = ((head)->tqh_first);                                \
                (var) != NULL && ((next) = TAILQ_NEXT((var), field), 1); \
                    (var) = (next))
#endif
#ifndef STAILQ_FOREACH_SAFE
#define STAILQ_FOREACH_SAFE(var, head, field, tvar)                      \
    for ((var) = STAILQ_FIRST((head));                                   \
        (var) && ((tvar) = STAILQ_NEXT((var), field), 1);                \
        (var) = (tvar))
#endif
#endif

/*
 * Override buggy implementations in sys/queue.h
 */
#undef TAILQ_REMOVE
#define TAILQ_REMOVE(head, elm, field) do {                \
    if ((elm) == (head)->tqh_first)                     \
        (head)->tqh_first = (elm)->field.tqe_next;      \
    if (((elm)->field.tqe_next) != NULL)                \
        (elm)->field.tqe_next->field.tqe_prev =         \
            (elm)->field.tqe_prev;                \
    else                                \
        (head)->tqh_last = (elm)->field.tqe_prev;        \
    *(elm)->field.tqe_prev = (elm)->field.tqe_next;            \
    (elm)->field.tqe_prev = NULL;                       \
} while (/*CONSTCOND*/0)


#endif
