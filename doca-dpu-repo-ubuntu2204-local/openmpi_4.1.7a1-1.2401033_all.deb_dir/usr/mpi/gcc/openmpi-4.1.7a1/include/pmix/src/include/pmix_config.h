/* src/include/pmix_config.h.  Generated from pmix_config.h.in by configure.  */
/* src/include/pmix_config.h.in.  Generated from configure.ac by autoheader.  */

/* -*- c -*-
 *
 * Copyright (c) 2004-2005 The Trustees of Indiana University.
 *                         All rights reserved.
 * Copyright (c) 2004-2005 The Trustees of the University of Tennessee.
 *                         All rights reserved.
 * Copyright (c) 2004-2005 High Performance Computing Center Stuttgart,
 *                         University of Stuttgart.  All rights reserved.
 * Copyright (c) 2004-2005 The Regents of the University of California.
 *                         All rights reserved.
 * Copyright (c) 2013-2015 Intel, Inc. All rights reserved
 * Copyright (c) 2016      IBM Corporation.  All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 *
 * This file is automatically generated by configure.  Edits will be lost
 * the next time you run configure!
 */

#ifndef PMIX_CONFIG_H
#define PMIX_CONFIG_H

#include "src/include/pmix_config_top.h"



/* Define if building universal (internal helper macro) */
/* #undef AC_APPLE_UNIVERSAL_BUILD */

/* Define to 1 if you have the <arpa/inet.h> header file. */
#define HAVE_ARPA_INET_H 1

/* Define to 1 if you have the `asprintf' function. */
#define HAVE_ASPRINTF 1

/* Define to 1 if you have the <crt_externs.h> header file. */
/* #undef HAVE_CRT_EXTERNS_H */

/* Define to 1 if you have the declaration of `AF_INET6', and to 0 if you
   don't. */
#define HAVE_DECL_AF_INET6 1

/* Define to 1 if you have the declaration of `AF_UNSPEC', and to 0 if you
   don't. */
#define HAVE_DECL_AF_UNSPEC 1

/* Define to 1 if you have the declaration of `PF_INET6', and to 0 if you
   don't. */
#define HAVE_DECL_PF_INET6 1

/* Define to 1 if you have the declaration of `PF_UNSPEC', and to 0 if you
   don't. */
#define HAVE_DECL_PF_UNSPEC 1

/* Define to 1 if you have the declaration of `__func__', and to 0 if you
   don't. */
#define HAVE_DECL___FUNC__ 1

/* Define to 1 if you have the <dirent.h> header file. */
#define HAVE_DIRENT_H 1

/* Define to 1 if you have the <dlfcn.h> header file. */
#define HAVE_DLFCN_H 1

/* Define to 1 if you have the <event.h> header file. */
/* #undef HAVE_EVENT_H */

/* Define to 1 if you have the `execve' function. */
#define HAVE_EXECVE 1

/* Define to 1 if you have the <fcntl.h> header file. */
#define HAVE_FCNTL_H 1

/* Define to 1 if you have the `fork' function. */
#define HAVE_FORK 1

/* Define to 1 if you have the `getpeereid' function. */
/* #undef HAVE_GETPEEREID */

/* Define to 1 if you have the `getpeerucred' function. */
/* #undef HAVE_GETPEERUCRED */

/* Define to 1 if you have the <grp.h> header file. */
#define HAVE_GRP_H 1

/* Define to 1 if you have the <hostLib.h> header file. */
/* #undef HAVE_HOSTLIB_H */

/* Define to 1 if you have the <hwloc.h> header file. */
/* #undef HAVE_HWLOC_H */

/* Define to 1 if you have the <ifaddrs.h> header file. */
#define HAVE_IFADDRS_H 1

/* Define to 1 if the system has the type `int16_t'. */
#define HAVE_INT16_T 1

/* Define to 1 if the system has the type `int32_t'. */
#define HAVE_INT32_T 1

/* Define to 1 if the system has the type `int64_t'. */
#define HAVE_INT64_T 1

/* Define to 1 if the system has the type `int8_t'. */
#define HAVE_INT8_T 1

/* Define to 1 if the system has the type `intptr_t'. */
#define HAVE_INTPTR_T 1

/* Define to 1 if you have the <inttypes.h> header file. */
#define HAVE_INTTYPES_H 1

/* Define to 1 if you have the <ioLib.h> header file. */
/* #undef HAVE_IOLIB_H */

/* Define to 1 if you have the `event_core' library (-levent_core). */
/* #undef HAVE_LIBEVENT_CORE */

/* Define to 1 if you have the `event_pthreads' library (-levent_pthreads). */
/* #undef HAVE_LIBEVENT_PTHREADS */

/* Define to 1 if you have the <libgen.h> header file. */
#define HAVE_LIBGEN_H 1

/* Define to 1 if you have the <libutil.h> header file. */
/* #undef HAVE_LIBUTIL_H */

/* Define to 1 if you have the <limits.h> header file. */
#define HAVE_LIMITS_H 1

/* Define to 1 if the system has the type `long long'. */
#define HAVE_LONG_LONG 1

/* Define to 1 if you have the <ltdl.h> header file. */
#define HAVE_LTDL_H 1

/* Define to 1 if you have the <memory.h> header file. */
#define HAVE_MEMORY_H 1

/* Define to 1 if you have the <munge.h> header file. */
/* #undef HAVE_MUNGE_H */

/* Define to 1 if you have the <netdb.h> header file. */
#define HAVE_NETDB_H 1

/* Define to 1 if you have the <netinet/in.h> header file. */
#define HAVE_NETINET_IN_H 1

/* Define to 1 if you have the <netinet/tcp.h> header file. */
#define HAVE_NETINET_TCP_H 1

/* Define to 1 if you have the <net/if.h> header file. */
#define HAVE_NET_IF_H 1

/* Define to 1 if you have the <net/uio.h> header file. */
/* #undef HAVE_NET_UIO_H */

/* Define to 1 if you have the `openpty' function. */
#define HAVE_OPENPTY 1

/* Define to 1 if you have the `posix_fallocate' function. */
#define HAVE_POSIX_FALLOCATE 1

/* Define to 1 if the system has the type `ptrdiff_t'. */
#define HAVE_PTRDIFF_T 1

/* Define to 1 if you have the `ptsname' function. */
#define HAVE_PTSNAME 1

/* Define to 1 if you have the <pty.h> header file. */
#define HAVE_PTY_H 1

/* Define to 1 if you have the `setenv' function. */
#define HAVE_SETENV 1

/* Define to 1 if you have the `setpgid' function. */
#define HAVE_SETPGID 1

/* Define to 1 if `si_band' is a member of `siginfo_t'. */
#define HAVE_SIGINFO_T_SI_BAND 1

/* Define to 1 if `si_fd' is a member of `siginfo_t'. */
#define HAVE_SIGINFO_T_SI_FD 1

/* Define to 1 if you have the <signal.h> header file. */
#define HAVE_SIGNAL_H 1

/* Define to 1 if you have the `snprintf' function. */
#define HAVE_SNPRINTF 1

/* Define to 1 if you have the `socketpair' function. */
#define HAVE_SOCKETPAIR 1

/* Define to 1 if the system has the type `socklen_t'. */
#define HAVE_SOCKLEN_T 1

/* Define to 1 if you have the <sockLib.h> header file. */
/* #undef HAVE_SOCKLIB_H */

/* Define to 1 if you have the `statfs' function. */
#define HAVE_STATFS 1

/* Define to 1 if you have the `statvfs' function. */
#define HAVE_STATVFS 1

/* Define to 1 if you have the <stdarg.h> header file. */
#define HAVE_STDARG_H 1

/* Define to 1 if you have the <stdbool.h> header file. */
#define HAVE_STDBOOL_H 1

/* Define to 1 if you have the <stddef.h> header file. */
#define HAVE_STDDEF_H 1

/* Define to 1 if you have the <stdint.h> header file. */
#define HAVE_STDINT_H 1

/* Define to 1 if you have the <stdlib.h> header file. */
#define HAVE_STDLIB_H 1

/* Define to 1 if you have the <strings.h> header file. */
#define HAVE_STRINGS_H 1

/* Define to 1 if you have the <string.h> header file. */
#define HAVE_STRING_H 1

/* Define to 1 if you have the `strncpy_s' function. */
/* #undef HAVE_STRNCPY_S */

/* Define to 1 if you have the `strnlen' function. */
#define HAVE_STRNLEN 1

/* Define to 1 if you have the <stropts.h> header file. */
/* #undef HAVE_STROPTS_H */

/* Define to 1 if you have the `strsignal' function. */
#define HAVE_STRSIGNAL 1

/* Define to 1 if `d_type' is a member of `struct dirent'. */
#define HAVE_STRUCT_DIRENT_D_TYPE 1

/* Define to 1 if `ifr_hwaddr' is a member of `struct ifreq'. */
#define HAVE_STRUCT_IFREQ_IFR_HWADDR 1

/* Define to 1 if `ifr_mtu' is a member of `struct ifreq'. */
#define HAVE_STRUCT_IFREQ_IFR_MTU 1

/* Define to 1 if the system has the type `struct sockaddr_in'. */
#define HAVE_STRUCT_SOCKADDR_IN 1

/* Define to 1 if the system has the type `struct sockaddr_in6'. */
#define HAVE_STRUCT_SOCKADDR_IN6 1

/* Define to 1 if `sa_len' is a member of `struct sockaddr'. */
/* #undef HAVE_STRUCT_SOCKADDR_SA_LEN */

/* Define to 1 if the system has the type `struct sockaddr_storage'. */
#define HAVE_STRUCT_SOCKADDR_STORAGE 1

/* Define to 1 if the system has the type `struct sockaddr_un'. */
#define HAVE_STRUCT_SOCKADDR_UN 1

/* Define to 1 if `uid' is a member of `struct sockpeercred'. */
/* #undef HAVE_STRUCT_SOCKPEERCRED_UID */

/* Define to 1 if `f_fstypename' is a member of `struct statfs'. */
/* #undef HAVE_STRUCT_STATFS_F_FSTYPENAME */

/* Define to 1 if `f_type' is a member of `struct statfs'. */
#define HAVE_STRUCT_STATFS_F_TYPE 1

/* Define to 1 if `f_basetype' is a member of `struct statvfs'. */
/* #undef HAVE_STRUCT_STATVFS_F_BASETYPE */

/* Define to 1 if `f_fstypename' is a member of `struct statvfs'. */
/* #undef HAVE_STRUCT_STATVFS_F_FSTYPENAME */

/* Define to 1 if `cr_uid' is a member of `struct ucred'. */
/* #undef HAVE_STRUCT_UCRED_CR_UID */

/* Define to 1 if `uid' is a member of `struct ucred'. */
#define HAVE_STRUCT_UCRED_UID 1

/* Define to 1 if you have the <syslog.h> header file. */
#define HAVE_SYSLOG_H 1

/* Define to 1 if you have the <sys/auxv.h> header file. */
#define HAVE_SYS_AUXV_H 1

/* Define to 1 if you have the <sys/cdefs.h> header file. */
#define HAVE_SYS_CDEFS_H 1

/* Define to 1 if you have the <sys/fcntl.h> header file. */
#define HAVE_SYS_FCNTL_H 1

/* Define to 1 if you have the <sys/ioctl.h> header file. */
#define HAVE_SYS_IOCTL_H 1

/* Define to 1 if you have the <sys/mount.h> header file. */
#define HAVE_SYS_MOUNT_H 1

/* Define to 1 if you have the <sys/param.h> header file. */
#define HAVE_SYS_PARAM_H 1

/* Define to 1 if you have the <sys/select.h> header file. */
#define HAVE_SYS_SELECT_H 1

/* Define to 1 if you have the <sys/socket.h> header file. */
#define HAVE_SYS_SOCKET_H 1

/* Define to 1 if you have the <sys/sockio.h> header file. */
/* #undef HAVE_SYS_SOCKIO_H */

/* Define to 1 if you have the <sys/statfs.h> header file. */
#define HAVE_SYS_STATFS_H 1

/* Define to 1 if you have the <sys/statvfs.h> header file. */
#define HAVE_SYS_STATVFS_H 1

/* Define to 1 if you have the <sys/stat.h> header file. */
#define HAVE_SYS_STAT_H 1

/* Define to 1 if you have the <sys/sysctl.h> header file. */
/* #undef HAVE_SYS_SYSCTL_H */

/* Define to 1 if you have the <sys/time.h> header file. */
#define HAVE_SYS_TIME_H 1

/* Define to 1 if you have the <sys/types.h> header file. */
#define HAVE_SYS_TYPES_H 1

/* Define to 1 if you have the <sys/uio.h> header file. */
#define HAVE_SYS_UIO_H 1

/* Define to 1 if you have the <sys/un.h> header file. */
#define HAVE_SYS_UN_H 1

/* Define to 1 if you have the <sys/utsname.h> header file. */
#define HAVE_SYS_UTSNAME_H 1

/* Define to 1 if you have the <sys/wait.h> header file. */
#define HAVE_SYS_WAIT_H 1

/* Define to 1 if you have the `tcgetpgrp' function. */
#define HAVE_TCGETPGRP 1

/* Define to 1 if you have the <termios.h> header file. */
#define HAVE_TERMIOS_H 1

/* Define to 1 if you have the <termio.h> header file. */
#define HAVE_TERMIO_H 1

/* Define to 1 if you have the <time.h> header file. */
#define HAVE_TIME_H 1

/* Define to 1 if you have the <ucred.h> header file. */
/* #undef HAVE_UCRED_H */

/* Define to 1 if the system has the type `uint128_t'. */
/* #undef HAVE_UINT128_T */

/* Define to 1 if the system has the type `uint16_t'. */
#define HAVE_UINT16_T 1

/* Define to 1 if the system has the type `uint32_t'. */
#define HAVE_UINT32_T 1

/* Define to 1 if the system has the type `uint64_t'. */
#define HAVE_UINT64_T 1

/* Define to 1 if the system has the type `uint8_t'. */
#define HAVE_UINT8_T 1

/* Define to 1 if the system has the type `uintptr_t'. */
#define HAVE_UINTPTR_T 1

/* Define to 1 if you have the <unistd.h> header file. */
#define HAVE_UNISTD_H 1

/* whether unix byteswap routines -- htonl, htons, nothl, ntohs -- are
   available */
#define HAVE_UNIX_BYTESWAP 1

/* Define to 1 if you have the `usleep' function. */
#define HAVE_USLEEP 1

/* Define to 1 if you have the <util.h> header file. */
/* #undef HAVE_UTIL_H */

/* Define to 1 if you have the <utmp.h> header file. */
#define HAVE_UTMP_H 1

/* Define to 1 if you have the `vasprintf' function. */
#define HAVE_VASPRINTF 1

/* Define to 1 if you have the `vsnprintf' function. */
#define HAVE_VSNPRINTF 1

/* Define to 1 if you have the `waitpid' function. */
#define HAVE_WAITPID 1

/* Define to 1 if you have the <zlib.h> header file. */
#define HAVE_ZLIB_H 1

/* Define to 1 if the system has the type `__int128'. */
#define HAVE___INT128 1

/* Define to the sub-directory where libtool stores uninstalled libraries. */
#define LT_OBJDIR ".libs/"

/* Define to the address where bug reports for this package should be sent. */
#define PACKAGE_BUGREPORT "https://github.com/pmix/pmix/issues"

/* Define to the full name of this package. */
#define PACKAGE_NAME "pmix"

/* Define to the full name and version of this package. */
#define PACKAGE_STRING "pmix 3.2.5a1"

/* Define to the one symbol short name of this package. */
#define PACKAGE_TARNAME "pmix"

/* Define to the home page for this package. */
#define PACKAGE_URL ""

/* Define to the version of this package. */
#define PACKAGE_VERSION "3.2.5a1"

/* Alignment of type bool */
#define PMIX_ALIGNMENT_BOOL 1

/* Alignment of type char */
#define PMIX_ALIGNMENT_CHAR 1

/* Alignment of type double */
#define PMIX_ALIGNMENT_DOUBLE 8

/* Alignment of type float */
#define PMIX_ALIGNMENT_FLOAT 4

/* Alignment of type int */
#define PMIX_ALIGNMENT_INT 4

/* Alignment of type int16_t */
#define PMIX_ALIGNMENT_INT16 2

/* Alignment of type int32_t */
#define PMIX_ALIGNMENT_INT32 4

/* Alignment of type int64_t */
#define PMIX_ALIGNMENT_INT64 8

/* Alignment of type int8_t */
#define PMIX_ALIGNMENT_INT8 1

/* Alignment of type long */
#define PMIX_ALIGNMENT_LONG 8

/* Alignment of type long double */
/* #undef PMIX_ALIGNMENT_LONG_DOUBLE */

/* Alignment of type long long */
#define PMIX_ALIGNMENT_LONG_LONG 8

/* Alignment of type short */
#define PMIX_ALIGNMENT_SHORT 2

/* Alignment of type size_t */
#define PMIX_ALIGNMENT_SIZE_T 8

/* Alignment of type void * */
#define PMIX_ALIGNMENT_VOID_P 8

/* Alignment of type wchar_t */
#define PMIX_ALIGNMENT_WCHAR 4

/* What ARM assembly version to use */
#define PMIX_ASM_ARM_VERSION 8

/* Whether we can do 64bit assembly operations or not. Should not be used
   outside of the assembly header files */
#define PMIX_ASM_SUPPORT_64BIT 1

/* Architecture type of assembly to use for atomic operations and CMA */
#define PMIX_ASSEMBLY_ARCH PMIX_ARM64

/* Whether to use builtin atomics */
#define PMIX_ASSEMBLY_BUILTIN PMIX_BUILTIN_C11

/* Whether we have support for RDTSCP instruction */
#define PMIX_ASSEMBLY_SUPPORTS_RDTSCP 0

/* The compiler $lower which PMIx was built with */
#define PMIX_BUILD_PLATFORM_COMPILER_FAMILYID 1

/* The compiler $lower which PMIX was built with */
#define PMIX_BUILD_PLATFORM_COMPILER_FAMILYNAME GNU

/* The compiler $lower which PMIx was built with */
#define PMIX_BUILD_PLATFORM_COMPILER_VERSION 721920

/* The compiler $lower which PMIx was built with */
#define PMIX_BUILD_PLATFORM_COMPILER_VERSION_STR 11.4.0

/* OMPI underlying C compiler */
#define PMIX_CC "gcc"

/* Capture the configure cmd line */
#define PMIX_CONFIGURE_CLI " \'--disable-man-pages\' \'--with-devel-headers\' \'--disable-debug\' \'--with-pmix-symbol-rename=OPAL_MCA_PMIX3X_\' \'--enable-embedded-mode\' \'--disable-pmix-timing\' \'--without-tests-examples\' \'--disable-pmix-binaries\' \'--disable-pmix-backward-compatibility\' \'--disable-visibility\' \'--enable-embedded-libevent\' '--with-libevent-header=\"opal/mca/event/libevent2022/libevent2022.h\"' \'--enable-embedded-hwloc\' '--with-hwloc-header=\"opal/mca/hwloc/hwloc201/hwloc201.h\"' \'--build=aarch64-linux-gnu\' \'--prefix=/usr\' \'--includedir=/usr/mpi/gcc/openmpi-4.1.7a1/include\' \'--mandir=/usr/mpi/gcc/openmpi-4.1.7a1/share/man\' \'--infodir=/usr/mpi/gcc/openmpi-4.1.7a1/share/info\' \'--sysconfdir=/etc\' \'--localstatedir=/var\' \'--disable-option-checking\' \'--disable-silent-rules\' \'--libexecdir=/usr/mpi/gcc/openmpi-4.1.7a1/lib/openmpi\' \'--disable-maintainer-mode\' \'--disable-dependency-tracking\' \'--prefix=/usr/mpi/gcc/openmpi-4.1.7a1\' \'build_alias=aarch64-linux-gnu\' \'CFLAGS=-g -ffile-prefix-map=/var/tmp/openmpi/openmpi-4.1.7a1=. -flto=auto -ffat-lto-objects -flto=auto -ffat-lto-objects -fstack-protector-strong -Wformat -Werror=format-security -O3\' \'LDFLAGS=-Wl,-Bsymbolic-functions -flto=auto -ffat-lto-objects -flto=auto -Wl,-z,relro\' \'CPPFLAGS=-Wdate-time -D_FORTIFY_SOURCE=2\' \'CXXFLAGS=-g -ffile-prefix-map=/var/tmp/openmpi/openmpi-4.1.7a1=. -flto=auto -ffat-lto-objects -flto=auto -ffat-lto-objects -fstack-protector-strong -Wformat -Werror=format-security -O3\' \'FCFLAGS=-g -ffile-prefix-map=/var/tmp/openmpi/openmpi-4.1.7a1=. -flto=auto -ffat-lto-objects -flto=auto -ffat-lto-objects -fstack-protector-strong -O3\' \'CFLAGS=-DNDEBUG -O3 -g \' \'CPPFLAGS=-I/var/tmp/openmpi/openmpi-4.1.7a1 -I/var/tmp/openmpi/openmpi-4.1.7a1 -I/var/tmp/openmpi/openmpi-4.1.7a1/opal/include -I/var/tmp/openmpi/openmpi-4.1.7a1/opal/include -I/var/tmp/openmpi/openmpi-4.1.7a1/opal/mca/event/libevent2022/libevent -I/var/tmp/openmpi/openmpi-4.1.7a1/opal/mca/event/libevent2022/libevent/include -I/var/tmp/openmpi/openmpi-4.1.7a1/opal/mca/hwloc/hwloc201/hwloc/include -Wdate-time -D_FORTIFY_SOURCE=2 \' \'--cache-file=/dev/null\' \'--srcdir=.\' \'--disable-option-checking\'"

/* Date when PMIx was built */
#define PMIX_CONFIGURE_DATE "2013-05-02T07:55:55"

/* Hostname where PMIx was built */
#define PMIX_CONFIGURE_HOST "reproduciblebuild"

/* User who built PMIx */
#define PMIX_CONFIGURE_USER "reproduciblebuild"

/* Whether C compiler supports GCC style inline assembly */
#define PMIX_C_GCC_INLINE_ASSEMBLY 1

/* Whether C compiler supports atomic convenience variables in stdatomic.h */
#define PMIX_C_HAVE_ATOMIC_CONV_VAR 1

/* Whether C compiler supports __builtin_clz */
#define PMIX_C_HAVE_BUILTIN_CLZ 1

/* Whether C compiler supports __builtin_expect */
#define PMIX_C_HAVE_BUILTIN_EXPECT 1

/* Whether C compiler supports __builtin_prefetch */
#define PMIX_C_HAVE_BUILTIN_PREFETCH 1

/* Whether C compiler supports __Atomic keyword */
#define PMIX_C_HAVE__ATOMIC 1

/* Whether C compiler supports __Generic keyword */
#define PMIX_C_HAVE__GENERIC 1

/* Whether C compiler supports _Static_assert keyword */
#define PMIX_C_HAVE__STATIC_ASSERT 1

/* Whether C compiler supports __Thread_local */
#define PMIX_C_HAVE__THREAD_LOCAL 1

/* Whether C compiler supports __thread */
#define PMIX_C_HAVE___THREAD 1

/* Whether we are in debugging mode or not */
#define PMIX_ENABLE_DEBUG 0

/* Whether we want to enable dlopen support */
#define PMIX_ENABLE_DLOPEN_SUPPORT 1

/* Whether we should enable thread support within the PMIX code base */
#define PMIX_ENABLE_MULTI_THREADS 1

/* Whether user wants PTY support or not */
#define PMIX_ENABLE_PTY_SUPPORT 1

/* Whether we want developer-level timing support or not */
#define PMIX_ENABLE_TIMING 0

/* Location of event2/thread.h */
#define PMIX_EVENT2_THREAD_HEADER "opal/mca/event/libevent2022/libevent2022.h"

/* Location of event.h */
#define PMIX_EVENT_HEADER "opal/mca/event/libevent2022/libevent2022.h"

/* Whether or not we have apple */
#define PMIX_HAVE_APPLE 0

/* Whether your compiler has __attribute__ or not */
#define PMIX_HAVE_ATTRIBUTE 1

/* Whether your compiler has __attribute__ aligned or not */
#define PMIX_HAVE_ATTRIBUTE_ALIGNED 1

/* Whether your compiler has __attribute__ always_inline or not */
#define PMIX_HAVE_ATTRIBUTE_ALWAYS_INLINE 1

/* Whether your compiler has __attribute__ cold or not */
#define PMIX_HAVE_ATTRIBUTE_COLD 1

/* Whether your compiler has __attribute__ const or not */
#define PMIX_HAVE_ATTRIBUTE_CONST 1

/* Whether your compiler has __attribute__ deprecated or not */
#define PMIX_HAVE_ATTRIBUTE_DEPRECATED 1

/* Whether your compiler has __attribute__ deprecated with optional argument
   */
#define PMIX_HAVE_ATTRIBUTE_DEPRECATED_ARGUMENT 1

/* Whether your compiler has __attribute__ destructor or not */
#define PMIX_HAVE_ATTRIBUTE_DESTRUCTOR 1

/* Whether your compiler has __attribute__ extension or not */
#define PMIX_HAVE_ATTRIBUTE_EXTENSION 1

/* Whether your compiler has __attribute__ format or not */
#define PMIX_HAVE_ATTRIBUTE_FORMAT 1

/* Whether your compiler has __attribute__ format and it works on function
   pointers */
#define PMIX_HAVE_ATTRIBUTE_FORMAT_FUNCPTR 1

/* Whether your compiler has __attribute__ hot or not */
#define PMIX_HAVE_ATTRIBUTE_HOT 1

/* Whether your compiler has __attribute__ malloc or not */
#define PMIX_HAVE_ATTRIBUTE_MALLOC 1

/* Whether your compiler has __attribute__ may_alias or not */
#define PMIX_HAVE_ATTRIBUTE_MAY_ALIAS 1

/* Whether your compiler has __attribute__ nonnull or not */
#define PMIX_HAVE_ATTRIBUTE_NONNULL 1

/* Whether your compiler has __attribute__ noreturn or not */
#define PMIX_HAVE_ATTRIBUTE_NORETURN 1

/* Whether your compiler has __attribute__ noreturn and it works on function
   pointers */
#define PMIX_HAVE_ATTRIBUTE_NORETURN_FUNCPTR 1

/* Whether your compiler has __attribute__ no_instrument_function or not */
#define PMIX_HAVE_ATTRIBUTE_NO_INSTRUMENT_FUNCTION 1

/* Whether your compiler has __attribute__ optnone or not */
#define PMIX_HAVE_ATTRIBUTE_OPTNONE 0

/* Whether your compiler has __attribute__ packed or not */
#define PMIX_HAVE_ATTRIBUTE_PACKED 1

/* Whether your compiler has __attribute__ pure or not */
#define PMIX_HAVE_ATTRIBUTE_PURE 1

/* Whether your compiler has __attribute__ sentinel or not */
#define PMIX_HAVE_ATTRIBUTE_SENTINEL 1

/* Whether your compiler has __attribute__ unused or not */
#define PMIX_HAVE_ATTRIBUTE_UNUSED 1

/* Whether your compiler has __attribute__ visibility or not */
#define PMIX_HAVE_ATTRIBUTE_VISIBILITY 1

/* Whether your compiler has __attribute__ warn unused result or not */
#define PMIX_HAVE_ATTRIBUTE_WARN_UNUSED_RESULT 1

/* Whether your compiler has __attribute__ weak alias or not */
#define PMIX_HAVE_ATTRIBUTE_WEAK_ALIAS 

/* whether qsort is broken or not */
#define PMIX_HAVE_BROKEN_QSORT 0

/* Whether C11 atomic compare swap is both supported and lock-free on 128-bit
   values */
#define PMIX_HAVE_C11_CSWAP_INT128 0

/* whether ceil is found and available */
#define PMIX_HAVE_CEIL 1

/* whether clock_gettime is found and available */
#define PMIX_HAVE_CLOCK_GETTIME 1

/* Whether the processor supports the cmpxchg16b instruction */
/* #undef PMIX_HAVE_CMPXCHG16B */

/* whether dirname is found and available */
#define PMIX_HAVE_DIRNAME 1

/* Whether the __atomic builtin atomic compare swap is both supported and
   lock-free on 128-bit values */
#define PMIX_HAVE_GCC_BUILTIN_CSWAP_INT128 0

/* whether gethostbyname is found and available */
#define PMIX_HAVE_GETHOSTBYNAME 1

/* Whether or not we have hwloc support */
#define PMIX_HAVE_HWLOC 1

/* Whether we are building against libev */
#define PMIX_HAVE_LIBEV 0

/* Whether we are building against libevent */
/* #undef PMIX_HAVE_LIBEVENT */

/* whether openpty is found and available */
#define PMIX_HAVE_OPENPTY 1

/* Whether the PMIX PDL framework is functional or not */
#define PMIX_HAVE_PDL_SUPPORT 1

/* If PTHREADS implementation supports PTHREAD_MUTEX_ERRORCHECK */
#define PMIX_HAVE_PTHREAD_MUTEX_ERRORCHECK 1

/* If PTHREADS implementation supports PTHREAD_MUTEX_ERRORCHECK_NP */
#define PMIX_HAVE_PTHREAD_MUTEX_ERRORCHECK_NP 1

/* Whether we have SA_RESTART in <signal.h> or not */
#define PMIX_HAVE_SA_RESTART 1

/* whether socket is found and available */
#define PMIX_HAVE_SOCKET 1

/* Whether or not we have solaris */
#define PMIX_HAVE_SOLARIS 0

/* Whether the __sync builtin atomic compare and swap supports 128-bit values
   */
#define PMIX_HAVE_SYNC_BUILTIN_CSWAP_INT128 1

/* Whether we have __va_copy or not */
#define PMIX_HAVE_UNDERSCORE_VA_COPY 1

/* Whether we have va_copy or not */
#define PMIX_HAVE_VA_COPY 1

/* Whether C compiler supports symbol visibility or not */
#define PMIX_HAVE_VISIBILITY 0

/* Whether or not we have zlib support */
#define PMIX_HAVE_ZLIB 1

/* Location of hwloc.h */
#define PMIX_HWLOC_HEADER "opal/mca/hwloc/hwloc201/hwloc201.h"

/* ident string for PMIX */
#define PMIX_IDENT_STRING ""

/* The library major version is always available, contrary to VERSION */
#define PMIX_MAJOR_VERSION 3

/* MCA cmd line identifier */
#define PMIX_MCA_CMD_LINE_ID "pmca"

/* MCA prefix string for envars */
#define PMIX_MCA_PREFIX "PMIX_MCA_"

/* The library minor version is always available, contrary to VERSION */
#define PMIX_MINOR_VERSION 2

/* Whether the C compiler supports "bool" without any other help (such as
   <stdbool.h>) */
#define PMIX_NEED_C_BOOL 1

/* Whether libraries can be configured with destructor functions */
#define PMIX_NO_LIB_DESTRUCTOR 0

/* package/branding string for PMIx */
#define PMIX_PACKAGE_STRING "PMIx reproduciblebuild@reproduciblebuild Distribution"

/* Whether we have lt_dladvise or not */
#define PMIX_PDL_PLIBLTDL_HAVE_LT_DLADVISE 1

/* Whether or not we found the optional write_nonrecursive_np flag */
#define PMIX_PTHREAD_RWLOCK_PREFER_WRITER_NONRECURSIVE_NP 0

/* type to use for ptrdiff_t */
#define PMIX_PTRDIFF_TYPE ptrdiff_t

/* The library release version is always available, contrary to VERSION */
#define PMIX_RELEASE_VERSION 5

/* Default value for mca_base_component_show_load_errors MCA variable */
#define PMIX_SHOW_LOAD_ERRORS_DEFAULT 0

/* Whether to use <stdbool.h> or not */
#define PMIX_USE_STDBOOL_H 1

/* The library version is always available, contrary to VERSION */
#define PMIX_VERSION "3.2.5a1"

/* Enable per-user config files */
#define PMIX_WANT_HOME_CONFIG_FILES 1

/* if want pretty-print stack trace feature */
#define PMIX_WANT_PRETTY_PRINT_STACKTRACE 1

/* The size of `char', as computed by sizeof. */
#define SIZEOF_CHAR 1

/* The size of `double', as computed by sizeof. */
#define SIZEOF_DOUBLE 8

/* The size of `float', as computed by sizeof. */
#define SIZEOF_FLOAT 4

/* The size of `int', as computed by sizeof. */
#define SIZEOF_INT 4

/* The size of `long', as computed by sizeof. */
#define SIZEOF_LONG 8

/* The size of `long long', as computed by sizeof. */
#define SIZEOF_LONG_LONG 8

/* The size of `pid_t', as computed by sizeof. */
#define SIZEOF_PID_T 4

/* The size of `ptrdiff_t', as computed by sizeof. */
#define SIZEOF_PTRDIFF_T 8

/* The size of `short', as computed by sizeof. */
#define SIZEOF_SHORT 2

/* The size of `size_t', as computed by sizeof. */
#define SIZEOF_SIZE_T 8

/* The size of `ssize_t', as computed by sizeof. */
/* #undef SIZEOF_SSIZE_T */

/* The size of `void *', as computed by sizeof. */
#define SIZEOF_VOID_P 8

/* The size of `wchar_t', as computed by sizeof. */
#define SIZEOF_WCHAR_T 4

/* The size of `_Bool', as computed by sizeof. */
#define SIZEOF__BOOL 1

/* Define to 1 if you have the ANSI C header files. */
#define STDC_HEADERS 1

/* Enable extensions on AIX 3, Interix.  */
#ifndef _ALL_SOURCE
# define _ALL_SOURCE 1
#endif
/* Enable GNU extensions on systems that have them.  */
#ifndef _GNU_SOURCE
# define _GNU_SOURCE 1
#endif
/* Enable threading extensions on Solaris.  */
#ifndef _POSIX_PTHREAD_SEMANTICS
# define _POSIX_PTHREAD_SEMANTICS 1
#endif
/* Enable extensions on HP NonStop.  */
#ifndef _TANDEM_SOURCE
# define _TANDEM_SOURCE 1
#endif
/* Enable general extensions on Solaris.  */
#ifndef __EXTENSIONS__
# define __EXTENSIONS__ 1
#endif


/* Define WORDS_BIGENDIAN to 1 if your processor stores words with the most
   significant byte first (like Motorola and SPARC, unlike Intel). */
#if defined AC_APPLE_UNIVERSAL_BUILD
# if defined __BIG_ENDIAN__
#  define WORDS_BIGENDIAN 1
# endif
#else
# ifndef WORDS_BIGENDIAN
/* #  undef WORDS_BIGENDIAN */
# endif
#endif

/* Define to 1 if `lex' declares `yytext' as a `char *' by default, not a
   `char[]'. */
#define YYTEXT_POINTER 1

/* Enable GNU extensions on systems that have them.  */
#ifndef _GNU_SOURCE
# define _GNU_SOURCE 1
#endif

/* Define to 1 if on MINIX. */
/* #undef _MINIX */

/* Define to 2 if the system does not provide POSIX.1 features except with
   this defined. */
/* #undef _POSIX_1_SOURCE */

/* Define to 1 if you need to in order for `stat' and other things to work. */
/* #undef _POSIX_SOURCE */

/* Define to `__inline__' or `__inline' if that's what the C compiler
   calls it, or to nothing if 'inline' is not supported under any name.  */
#ifndef __cplusplus
#define inline __inline__
#endif


#include "src/include/pmix_config_bottom.h"
#endif /* PMIX_CONFIG_H */

