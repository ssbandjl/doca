/**
* Copyright (c) 2001-2011, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
* This software product is a proprietary product of Mellanox Technologies Ltd.
* (the "Company") and all right, title, and interest and to the software product,
* including all associated intellectual property rights, are and shall
* remain exclusively with the Company.
*
* This software product is governed by the End User License Agreement
* provided with the software product.
* $COPYRIGHT$
* $HEADER$
*/


#ifndef __HCOLL_VERSION_H__
#define __HCOLL_VERSION_H__

#define HCOLL_VERNO_MAJOR 4
#define HCOLL_VERNO_MINOR 8
#define HCOLL_VERNO_MICRO 3227
#define HCOLL_VERNO_STRING "4.8.3227"
#define HCOLL_GIT_SHA     "fa284ed"

#define HCOLL_MINOR_BIT   (16UL)
#define HCOLL_MAJOR_BIT   (24UL)
#define HCOLL_API         ((4L<<HCOLL_MAJOR_BIT)|(8L << HCOLL_MINOR_BIT))

#endif
