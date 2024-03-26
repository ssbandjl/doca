#ifndef _NVMX_UTIL_H
#define _NVMX_UTIL_H

#include <stdio.h>
#include <stdlib.h>

static inline uint32_t
nvmx_u32log2(uint32_t x)
{
	if (x == 0) {
		/* log(0) is undefined */
		return 0;
	}
	return 31u - __builtin_clz(x);
}


#endif
