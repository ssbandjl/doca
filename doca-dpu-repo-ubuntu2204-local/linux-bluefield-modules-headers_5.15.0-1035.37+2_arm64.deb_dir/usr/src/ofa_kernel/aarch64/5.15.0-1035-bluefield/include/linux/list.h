#ifndef _COMPAT_LINUX_LIST_H
#define _COMPAT_LINUX_LIST_H

/* Include the autogenerated header file */
#include "../../compat/config.h"

#include_next <linux/list.h>

#ifndef HAVE_LIST_IS_FIRST
static inline int list_is_first(const struct list_head *list, const struct list_head *head)
{
         return list->prev == head;
}
#endif

#define compat_hlist_for_each_entry_safe(pos, n, head, member)	\
	hlist_for_each_entry_safe(pos, n, head, member)

#define compat_hlist_for_each_entry(pos, head, member)		\
	hlist_for_each_entry(pos, head, member)

#define COMPAT_HL_NODE

#ifndef list_prev_entry
#define list_prev_entry(pos, member) \
	list_entry((pos)->member.prev, typeof(*(pos)), member)
#endif

#ifndef list_next_entry
#define list_next_entry(pos, member) \
	list_entry((pos)->member.next, typeof(*(pos)), member)
#endif

#ifndef list_first_entry_or_null
#define list_first_entry_or_null(ptr, type, member) \
	(!list_empty(ptr) ? list_first_entry(ptr, type, member) : NULL)
#endif

#endif /* _COMPAT_LINUX_LIST_H */
