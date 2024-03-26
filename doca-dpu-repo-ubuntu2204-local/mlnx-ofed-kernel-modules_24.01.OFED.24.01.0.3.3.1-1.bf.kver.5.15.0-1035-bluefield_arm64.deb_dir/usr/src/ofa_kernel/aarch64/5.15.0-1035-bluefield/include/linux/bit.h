#ifndef _COMPAT_LINUX_BIT_H
#define _COMPAT_LINUX_BIT_H

/* Include the autogenerated header file */
#include "../../compat/config.h"

#include_next <linux/bit.h>

#ifndef GENMASK 
#define GENMASK(h, l) \
	(((~0UL) - (1UL << (l)) + 1) & (~0UL >> (BITS_PER_LONG - 1 - (h))))
#endif

#endif /* _COMPAT_LINUX_BIT_H */
