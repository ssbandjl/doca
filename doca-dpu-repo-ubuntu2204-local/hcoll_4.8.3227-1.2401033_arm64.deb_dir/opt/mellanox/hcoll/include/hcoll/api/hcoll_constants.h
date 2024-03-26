/*
 * Copyright (c) 2004-2005 The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2004-2005 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2004-2005 High Performance Computing Center Stuttgart, 
 *                         University of Stuttgart.  All rights reserved.
 * Copyright (c) 2004-2005 The Regents of the University of California.
 *                         All rights reserved.
 * Copyright (c) 2010-2012 Cisco Systems, Inc.  All rights reserved.
 * $COPYRIGHT$
 * 
 * Additional copyrights may follow
 * 
 * $HEADER$
 */

#ifndef HCOLL_CONSTANTS_H
#define HCOLL_CONSTANTS_H
 
/* error codes - don't forget to update HCOLL/rutime/HCOLL_init.c when 
   adding to this list */
#define HCOLL_ERR_BASE             0 /* internal use only */
 
enum {
    HCOLL_SUCCESS                            = (HCOLL_ERR_BASE),

    HCOLL_ERROR                              = (HCOLL_ERR_BASE -  1),
    HCOLL_ERR_OUT_OF_RESOURCE                = (HCOLL_ERR_BASE -  2), /* fatal error */
    HCOLL_ERR_TEMP_OUT_OF_RESOURCE           = (HCOLL_ERR_BASE -  3), /* try again later */
    HCOLL_ERR_RESOURCE_BUSY                  = (HCOLL_ERR_BASE -  4),
    HCOLL_ERR_BAD_PARAM                      = (HCOLL_ERR_BASE -  5),  /* equivalent to MPI_ERR_ARG error code */
    HCOLL_ERR_FATAL                          = (HCOLL_ERR_BASE -  6),
    HCOLL_ERR_NOT_IMPLEMENTED                = (HCOLL_ERR_BASE -  7),
    HCOLL_ERR_NOT_SUPPORTED                  = (HCOLL_ERR_BASE -  8),
    HCOLL_ERR_INTERUPTED                     = (HCOLL_ERR_BASE -  9),
    HCOLL_ERR_WOULD_BLOCK                    = (HCOLL_ERR_BASE - 10),
    HCOLL_ERR_IN_ERRNO                       = (HCOLL_ERR_BASE - 11),
    HCOLL_ERR_UNREACH                        = (HCOLL_ERR_BASE - 12),
    HCOLL_ERR_NOT_FOUND                      = (HCOLL_ERR_BASE - 13),
    HCOLL_EXISTS                             = (HCOLL_ERR_BASE - 14), /* indicates that the specified object already exists */
    HCOLL_ERR_TIMEOUT                        = (HCOLL_ERR_BASE - 15),
    HCOLL_ERR_NOT_AVAILABLE                  = (HCOLL_ERR_BASE - 16),
    HCOLL_ERR_PERM                           = (HCOLL_ERR_BASE - 17), /* no permission */
    HCOLL_ERR_VALUE_OUT_OF_BOUNDS            = (HCOLL_ERR_BASE - 18),
    HCOLL_ERR_FILE_READ_FAILURE              = (HCOLL_ERR_BASE - 19),
    HCOLL_ERR_FILE_WRITE_FAILURE             = (HCOLL_ERR_BASE - 20),
    HCOLL_ERR_FILE_OPEN_FAILURE              = (HCOLL_ERR_BASE - 21),
    HCOLL_ERR_PACK_MISMATCH                  = (HCOLL_ERR_BASE - 22),
    HCOLL_ERR_PACK_FAILURE                   = (HCOLL_ERR_BASE - 23),
    HCOLL_ERR_UNPACK_FAILURE                 = (HCOLL_ERR_BASE - 24),
    HCOLL_ERR_UNPACK_INADEQUATE_SPACE        = (HCOLL_ERR_BASE - 25),
    HCOLL_ERR_UNPACK_READ_PAST_END_OF_BUFFER = (HCOLL_ERR_BASE - 26),
    HCOLL_ERR_TYPE_MISMATCH                  = (HCOLL_ERR_BASE - 27),
    HCOLL_ERR_OPERATION_UNSUPPORTED          = (HCOLL_ERR_BASE - 28),
    HCOLL_ERR_UNKNOWN_DATA_TYPE              = (HCOLL_ERR_BASE - 29),
    HCOLL_ERR_BUFFER                         = (HCOLL_ERR_BASE - 30),
    HCOLL_ERR_DATA_TYPE_REDEF                = (HCOLL_ERR_BASE - 31),
    HCOLL_ERR_DATA_OVERWRITE_ATTEMPT         = (HCOLL_ERR_BASE - 32),
    HCOLL_ERR_MODULE_NOT_FOUND               = (HCOLL_ERR_BASE - 33),
    HCOLL_ERR_TOPO_SLOT_LIST_NOT_SUPPORTED   = (HCOLL_ERR_BASE - 34),
    HCOLL_ERR_TOPO_SOCKET_NOT_SUPPORTED      = (HCOLL_ERR_BASE - 35),
    HCOLL_ERR_TOPO_CORE_NOT_SUPPORTED        = (HCOLL_ERR_BASE - 36),
    HCOLL_ERR_NOT_ENOUGH_SOCKETS             = (HCOLL_ERR_BASE - 37),
    HCOLL_ERR_NOT_ENOUGH_CORES               = (HCOLL_ERR_BASE - 38),
    HCOLL_ERR_INVALID_PHYS_CPU               = (HCOLL_ERR_BASE - 39),
    HCOLL_ERR_MULTIPLE_AFFINITIES            = (HCOLL_ERR_BASE - 40),
    HCOLL_ERR_SLOT_LIST_RANGE                = (HCOLL_ERR_BASE - 41),
    HCOLL_ERR_NETWORK_NOT_PARSEABLE          = (HCOLL_ERR_BASE - 42),
    HCOLL_ERR_SILENT                         = (HCOLL_ERR_BASE - 43),
    HCOLL_ERR_NOT_INITIALIZED                = (HCOLL_ERR_BASE - 44),
    HCOLL_ERR_NO_MATCH_YET                   = (HCOLL_ERR_BASE - 45)
};

#define HCOLL_ERR_MAX                (HCOLL_ERR_BASE - 100)
#endif /* HCOLL_CONSTANTS_H */

