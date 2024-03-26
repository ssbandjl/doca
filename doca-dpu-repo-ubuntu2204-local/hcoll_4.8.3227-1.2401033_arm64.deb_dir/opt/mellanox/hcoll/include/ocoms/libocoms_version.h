/*
 * Copyright (c) 2004-2008 The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2004-2005 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2004-2005 High Performance Computing Center Stuttgart,
 *                         University of Stuttgart.  All rights reserved.
 * Copyright (c) 2004-2005 The Regents of the University of California.
 *                         All rights reserved.
 * Copyright (c) 2008      Cisco Systems, Inc.  All rights reserved.
 * Copyright (c) 2011-2013 UT-Battelle, LLC. All rights reserved.
 * Copyright (C) 2013      Mellanox Technologies Ltd. All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */
 
#ifndef __LIBOCOMS_VERSION_H__
#define __LIBOCOMS_VERSION_H__

#define LIBOCOMS_MINOR_BIT   (16UL)
#define LIBOCOMS_MAJOR_BIT   (24UL)
#define LIBOCOMS_VERNO_MICRO svnversion
#define LIBOCOMS_API         (10)

#define LIBOCOMS_VERNO_STRING "1.0.svnversion"
#define LIBOCOMS_VER(major, minor) (((major)<<LIBOCOMS_MAJOR_BIT)|((minor)<<LIBOCOMS_MINOR_BIT))

#endif
