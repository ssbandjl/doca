/*-
 *   BSD LICENSE
 *
 *   Copyright (c) Intel Corporation.
 *   All rights reserved.
 *
 *   Redistribution and use in source and binary forms, with or without
 *   modification, are permitted provided that the following conditions
 *   are met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in
 *       the documentation and/or other materials provided with the
 *       distribution.
 *     * Neither the name of Intel Corporation nor the names of its
 *       contributors may be used to endorse or promote products derived
 *       from this software without specific prior written permission.
 *
 *   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 *   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 *   OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 *   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 *   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 *   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef NVME_UTF_H_
#define NVME_UTF_H_

#include <assert.h>
#include <ctype.h>
#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#include <arpa/inet.h>
#include <dirent.h>
#include <fcntl.h>
#include <ifaddrs.h>
#include <netdb.h>
#include <poll.h>
#include <pthread.h>
#include <semaphore.h>
#include <signal.h>
#include <syslog.h>
#include <termios.h>
#include <unistd.h>
#include <net/if.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/uio.h>
#include <sys/un.h>
#include <sys/user.h>
#include <sys/wait.h>

static inline bool
utf8_tail(uint8_t c)
{
	/* c >= 0x80 && c <= 0xBF, or binary 01xxxxxx */
	return (c & 0xC0) == 0x80;
}

/*
 * Check for a valid UTF-8 encoding of a single codepoint.
 *
 * \return Length of valid UTF-8 byte sequence, or negative if invalid.
 */
static inline int
utf8_valid(const uint8_t *start, const uint8_t *end)
{
	const uint8_t *p = start;
	uint8_t b0, b1, b2, b3;

	if (p == end) {
		return 0;
	}

	b0 = *p;

	if (b0 <= 0x7F) {
		return 1;
	}

	if (b0 <= 0xC1) {
		/* Invalid start byte */
		return -1;
	}

	if (++p == end) {
		/* Not enough bytes left */
		return -1;
	}
	b1 = *p;

	if (b0 <= 0xDF) {
		/* C2..DF 80..BF */
		if (!utf8_tail(b1)) {
			return -1;
		}
		return 2;
	}

	if (++p == end) {
		/* Not enough bytes left */
		return -1;
	}
	b2 = *p;

	if (b0 == 0xE0) {
		/* E0 A0..BF 80..BF */
		if (b1 < 0xA0 || b1 > 0xBF || !utf8_tail(b2)) {
			return -1;
		}
		return 3;
	} else if (b0 == 0xED && b1 >= 0xA0) {
		/*
		 * UTF-16 surrogate pairs use U+D800..U+DFFF, which would be encoded as
		 * ED A0..BF 80..BF in UTF-8; however, surrogate pairs are not allowed in UTF-8.
		 */
		return -1;
	} else if (b0 <= 0xEF) {
		/* E1..EF 80..BF 80..BF */
		if (!utf8_tail(b1) || !utf8_tail(b2)) {
			return -1;
		}
		return 3;
	}

	if (++p == end) {
		/* Not enough bytes left */
		return -1;
	}
	b3 = *p;

	if (b0 == 0xF0) {
		/* F0 90..BF 80..BF 80..BF */
		if (b1 < 0x90 || b1 > 0xBF || !utf8_tail(b2) || !utf8_tail(b3)) {
			return -1;
		}
		return 4;
	} else if (b0 <= 0xF3) {
		/* F1..F3 80..BF 80..BF 80..BF */
		if (!utf8_tail(b1) || !utf8_tail(b2) || !utf8_tail(b3)) {
			return -1;
		}
		return 4;
	} else if (b0 == 0xF4) {
		/* F4 80..8F 80..BF 80..BF */
		if (b1 < 0x80 || b1 > 0x8F || !utf8_tail(b2) || !utf8_tail(b3)) {
			return -1;
		}
		return 4;
	}

	return -1;
}

static inline uint32_t
utf8_decode_unsafe_1(const uint8_t *data)
{
	return data[0];
}

static inline uint32_t
utf8_decode_unsafe_2(const uint8_t *data)
{
	uint32_t codepoint;

	codepoint = ((data[0] & 0x1F) << 6);
	codepoint |= (data[1] & 0x3F);

	return codepoint;
}

static inline uint32_t
utf8_decode_unsafe_3(const uint8_t *data)
{
	uint32_t codepoint;

	codepoint = ((data[0] & 0x0F) << 12);
	codepoint |= (data[1] & 0x3F) << 6;
	codepoint |= (data[2] & 0x3F);

	return codepoint;
}

static inline uint32_t
utf8_decode_unsafe_4(const uint8_t *data)
{
	uint32_t codepoint;

	codepoint = ((data[0] & 0x07) << 18);
	codepoint |= (data[1] & 0x3F) << 12;
	codepoint |= (data[2] & 0x3F) << 6;
	codepoint |= (data[3] & 0x3F);

	return codepoint;
}

/*
 * Encode a single Unicode codepoint as UTF-8.
 *
 * buf must have at least 4 bytes of space available (hence unsafe).
 *
 * \return Number of bytes appended to buf, or negative if encoding failed.
 */
static inline int
utf8_encode_unsafe(uint8_t *buf, uint32_t c)
{
	if (c <= 0x7F) {
		buf[0] = c;
		return 1;
	} else if (c <= 0x7FF) {
		buf[0] = 0xC0 | (c >> 6);
		buf[1] = 0x80 | (c & 0x3F);
		return 2;
	} else if (c >= 0xD800 && c <= 0xDFFF) {
		/* UTF-16 surrogate pairs - invalid in UTF-8 */
		return -1;
	} else if (c <= 0xFFFF) {
		buf[0] = 0xE0 | (c >> 12);
		buf[1] = 0x80 | ((c >> 6) & 0x3F);
		buf[2] = 0x80 | (c & 0x3F);
		return 3;
	} else if (c <= 0x10FFFF) {
		buf[0] = 0xF0 | (c >> 18);
		buf[1] = 0x80 | ((c >> 12) & 0x3F);
		buf[2] = 0x80 | ((c >> 6) & 0x3F);
		buf[3] = 0x80 | (c & 0x3F);
		return 4;
	}
	return -1;
}

static inline int
utf8_codepoint_len(uint32_t c)
{
	if (c <= 0x7F) {
		return 1;
	} else if (c <= 0x7FF) {
		return 2;
	} else if (c >= 0xD800 && c <= 0xDFFF) {
		/* UTF-16 surrogate pairs - invalid in UTF-8 */
		return -1;
	} else if (c <= 0xFFFF) {
		return 3;
	} else if (c <= 0x10FFFF) {
		return 4;
	}
	return -1;
}

static inline bool
utf16_valid_surrogate_high(uint32_t val)
{
	return val >= 0xD800 && val <= 0xDBFF;
}

static inline bool
utf16_valid_surrogate_low(uint32_t val)
{
	return val >= 0xDC00 && val <= 0xDFFF;
}

static inline uint32_t
utf16_decode_surrogate_pair(uint32_t high, uint32_t low)
{
	uint32_t codepoint;

	assert(utf16_valid_surrogate_high(high));
	assert(utf16_valid_surrogate_low(low));

	codepoint = low;
	codepoint &= 0x3FF;
	codepoint |= ((high & 0x3FF) << 10);
	codepoint += 0x10000;

	return codepoint;
}

#endif
