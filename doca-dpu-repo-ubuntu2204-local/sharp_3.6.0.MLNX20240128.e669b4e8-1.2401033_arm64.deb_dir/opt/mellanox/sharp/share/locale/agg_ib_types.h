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

#ifndef AGG_IB_TYPES_H_
#define AGG_IB_TYPES_H_

#define IB_LID_UNASSIGNED            0
#define IB_MAX_PHYS_NUM_PORTS        254
#define IBNODE_UNASSIGNED_RANK       0xFF
#define FABRIC_MAX_VALID_LID         0xBFFF
#define IB_DEFAULT_SUBNET_PREFIX_STR "0xFE80000000000000"
#define IB_DEFAULT_SUBNET_PREFIX     (0xFE80000000000000ULL)

// ENUMS

// We only recognize CA or SW nodes
typedef enum
{
    IB_UNKNOWN_NODE_TYPE,
    IB_CA_NODE,
    IB_SW_NODE
} NodeType;
// IB_CA_NODE is ComputeNode or AggNode (AggNode a virtual node, it's basically the 'Sharp' chip on the switch)
// IB_SW_NODE is the actual switch

static inline NodeType char2nodetype(const char* w)
{
    if (!w || (*w == '\0'))
        return IB_UNKNOWN_NODE_TYPE;
    if (!strcmp(w, "SW"))
        return IB_SW_NODE;
    if (!strcmp(w, "CA"))
        return IB_CA_NODE;
    return IB_UNKNOWN_NODE_TYPE;
};

static inline const char* nodetype2char(const NodeType w)
{
    switch (w) {
        case IB_SW_NODE:
            return ("SW");
        case IB_CA_NODE:
            return ("CA");
        default:
            return ("UNKNOWN");
    }
};

typedef enum
{
    IB_UNKNOWN_LINK_WIDTH = 0,
    IB_LINK_WIDTH_1X = 1,
    IB_LINK_WIDTH_4X = 2,
    IB_LINK_WIDTH_8X = 4,
    IB_LINK_WIDTH_12X = 8,
} IBLinkWidth;

static inline IBLinkWidth char2width(const char* w)
{
    if (!w || (*w == '\0'))
        return IB_UNKNOWN_LINK_WIDTH;
    if (!strcmp(w, "1x"))
        return IB_LINK_WIDTH_1X;
    if (!strcmp(w, "4x"))
        return IB_LINK_WIDTH_4X;
    if (!strcmp(w, "8x"))
        return IB_LINK_WIDTH_8X;
    if (!strcmp(w, "12x"))
        return IB_LINK_WIDTH_12X;
    return IB_UNKNOWN_LINK_WIDTH;
};

static inline const char* width2char(const IBLinkWidth w)
{
    switch (w) {
        case IB_LINK_WIDTH_1X:
            return ("1x");
        case IB_LINK_WIDTH_4X:
            return ("4x");
        case IB_LINK_WIDTH_8X:
            return ("8x");
        case IB_LINK_WIDTH_12X:
            return ("12x");
        default:
            return ("UNKNOWN");
    }
};

typedef enum
{
    IB_UNKNOWN_LINK_SPEED = 0,
    IB_LINK_SPEED_2_5 = 1,
    IB_LINK_SPEED_5 = 2,
    IB_LINK_SPEED_10 = 4,
    IB_LINK_SPEED_14 = 1 << 8,      /* second byte is for extended ones */
    IB_LINK_SPEED_25 = 2 << 8,      /* second byte is for extended ones */
    IB_LINK_SPEED_FDR_10 = 1 << 16, /* third byte is for vendor specific ones */
    IB_LINK_SPEED_EDR_20 = 2 << 16  /* third byte is for vendor specific ones */
} IBLinkSpeed;

#define IB_UNKNOWN_LINK_SPEED_STR "UNKNOWN"
#define IB_LINK_SPEED_2_5_STR     "2.5"
#define IB_LINK_SPEED_5_STR       "5"
#define IB_LINK_SPEED_10_STR      "10"
#define IB_LINK_SPEED_14_STR      "14"
#define IB_LINK_SPEED_25_STR      "25"
#define IB_LINK_SPEED_FDR_10_STR  "FDR10"
#define IB_LINK_SPEED_EDR_20_STR  "EDR20"

static inline const char* speed2char(const IBLinkSpeed s)
{
    switch (s) {
        case IB_LINK_SPEED_2_5:
            return (IB_LINK_SPEED_2_5_STR);
        case IB_LINK_SPEED_5:
            return (IB_LINK_SPEED_5_STR);
        case IB_LINK_SPEED_10:
            return (IB_LINK_SPEED_10_STR);
        case IB_LINK_SPEED_14:
            return (IB_LINK_SPEED_14_STR);
        case IB_LINK_SPEED_25:
            return (IB_LINK_SPEED_25_STR);
        case IB_LINK_SPEED_FDR_10:
            return (IB_LINK_SPEED_FDR_10_STR);
        case IB_LINK_SPEED_EDR_20:
            return (IB_LINK_SPEED_EDR_20_STR);
        default:
            return (IB_UNKNOWN_LINK_SPEED_STR);
    }
};

static inline IBLinkSpeed char2speed(const char* s)
{
    if (!s || (*s == '\0'))
        return IB_UNKNOWN_LINK_SPEED;
    if (!strcmp(s, IB_LINK_SPEED_2_5_STR))
        return IB_LINK_SPEED_2_5;
    if (!strcmp(s, IB_LINK_SPEED_5_STR))
        return IB_LINK_SPEED_5;
    if (!strcmp(s, IB_LINK_SPEED_10_STR))
        return IB_LINK_SPEED_10;
    if (!strcmp(s, IB_LINK_SPEED_14_STR))
        return IB_LINK_SPEED_14;
    if (!strcmp(s, IB_LINK_SPEED_25_STR))
        return IB_LINK_SPEED_25;
    if (!strcmp(s, IB_LINK_SPEED_FDR_10_STR))
        return IB_LINK_SPEED_FDR_10;
    if (!strcmp(s, IB_LINK_SPEED_EDR_20_STR))
        return IB_LINK_SPEED_EDR_20;
    return IB_UNKNOWN_LINK_SPEED;
};

typedef enum
{
    IB_UNKNOWN_PORT_STATE = 0,
    IB_PORT_STATE_DOWN = 1,
    IB_PORT_STATE_INIT = 2,
    IB_PORT_STATE_ARM = 3,
    IB_PORT_STATE_ACTIVE = 4
} PortState;

static inline PortState char2portstate(const char* w)
{
    if (!w || (*w == '\0'))
        return IB_UNKNOWN_PORT_STATE;
    if (!strcmp(w, "DOWN"))
        return IB_PORT_STATE_DOWN;
    if (!strcmp(w, "INI"))
        return IB_PORT_STATE_INIT;
    if (!strcmp(w, "ARM"))
        return IB_PORT_STATE_ARM;
    if (!strcmp(w, "ACT"))
        return IB_PORT_STATE_ACTIVE;
    return IB_UNKNOWN_PORT_STATE;
};

static inline const char* portstate2char(const PortState w)
{
    switch (w) {
        case IB_PORT_STATE_DOWN:
            return ("DOWN");
        case IB_PORT_STATE_INIT:
            return ("INI");
        case IB_PORT_STATE_ARM:
            return ("ARM");
        case IB_PORT_STATE_ACTIVE:
            return ("ACT");
        default:
            return ("UNKNOWN");
    }
};

#endif   // AGG_IB_TYPES_H_
