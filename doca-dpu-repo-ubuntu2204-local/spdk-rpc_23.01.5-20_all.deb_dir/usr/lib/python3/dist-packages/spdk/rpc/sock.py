#  SPDX-License-Identifier: BSD-3-Clause
#  Copyright (C) 2021 Intel Corporation.
#  All rights reserved.


def sock_impl_get_options(client, impl_name=None):
    """Get parameters for the socket layer implementation.

    Args:
        impl_name: name of socket implementation, e.g. posix
    """
    params = {}

    params['impl_name'] = impl_name

    return client.call('sock_impl_get_options', params)


def sock_impl_set_options(client,
                          impl_name=None,
                          recv_buf_size=None,
                          send_buf_size=None,
                          enable_recv_pipe=None,
                          enable_quickack=None,
                          enable_placement_id=None,
                          enable_zerocopy_send_server=None,
                          enable_zerocopy_send_client=None,
                          zerocopy_threshold=None,
                          flush_batch_timeout=None,
                          flush_batch_iovcnt_threshold=None,
                          flush_batch_bytes_threshold=None,
                          tls_version=None,
                          enable_ktls=None,
                          psk_key=None,
                          psk_identity=None,
                          enable_zerocopy_recv=None,
                          enable_tcp_nodelay=None,
                          buffers_pool_size=None,
                          packets_pool_size=None,
                          enable_early_init=None):
    """Set parameters for the socket layer implementation.

    Args:
        impl_name: name of socket implementation, e.g. posix
        recv_buf_size: size of socket receive buffer in bytes (optional)
        send_buf_size: size of socket send buffer in bytes (optional)
        enable_recv_pipe: enable or disable receive pipe (optional)
        enable_quickack: enable or disable quickack (optional)
        enable_placement_id: option for placement_id. 0:disable,1:incoming_napi,2:incoming_cpu (optional)
        enable_zerocopy_send_server: enable or disable zerocopy on send for server sockets(optional)
        enable_zerocopy_send_client: enable or disable zerocopy on send for client sockets(optional)
        zerocopy_threshold: set zerocopy_threshold in bytes(optional)
        flush_batch_timeout: set flush_batch_timeout(optional)
        flush_batch_iovcnt_threshold: set flush_batch_iovcnt_threshold(optional)
        flush_batch_bytes_threshold: set flush_batch_bytes_threshold(optional)
        tls_version: set TLS protocol version (optional)
        enable_ktls: enable or disable Kernel TLS (optional)
        psk_key: set psk_key (optional)
        psk_identity: set psk_identity (optional)
        enable_zerocopy_recv: enable or disable zerocopy on receive (optional)
        enable_tcp_nodelay: enable or disable TCP_NODELAY socket option (optional)
        buffers_pool_size: per poll group socket buffers pool size (optional)
        packets_pool_size: per poll group packets pool size (optional)
        enable_early_init: enable or disable early initialization (optional)
    """
    params = {}

    params['impl_name'] = impl_name
    if recv_buf_size is not None:
        params['recv_buf_size'] = recv_buf_size
    if send_buf_size is not None:
        params['send_buf_size'] = send_buf_size
    if enable_recv_pipe is not None:
        params['enable_recv_pipe'] = enable_recv_pipe
    if enable_quickack is not None:
        params['enable_quickack'] = enable_quickack
    if enable_placement_id is not None:
        params['enable_placement_id'] = enable_placement_id
    if enable_zerocopy_send_server is not None:
        params['enable_zerocopy_send_server'] = enable_zerocopy_send_server
    if enable_zerocopy_send_client is not None:
        params['enable_zerocopy_send_client'] = enable_zerocopy_send_client
    if zerocopy_threshold is not None:
        params['zerocopy_threshold'] = zerocopy_threshold
    if flush_batch_timeout is not None:
        params['flush_batch_timeout'] = flush_batch_timeout
    if flush_batch_iovcnt_threshold is not None:
        params['flush_batch_iovcnt_threshold'] = flush_batch_iovcnt_threshold
    if flush_batch_bytes_threshold is not None:
        params['flush_batch_bytes_threshold'] = flush_batch_bytes_threshold
    if tls_version is not None:
        params['tls_version'] = tls_version
    if enable_ktls is not None:
        params['enable_ktls'] = enable_ktls
    if psk_key is not None:
        params['psk_key'] = psk_key
    if psk_identity is not None:
        params['psk_identity'] = psk_identity
    if enable_zerocopy_recv is not None:
        params['enable_zerocopy_recv'] = enable_zerocopy_recv
    if enable_tcp_nodelay is not None:
        params['enable_tcp_nodelay'] = enable_tcp_nodelay
    if buffers_pool_size is not None:
        params['buffers_pool_size'] = buffers_pool_size
    if packets_pool_size is not None:
        params['packets_pool_size'] = packets_pool_size
    if enable_early_init is not None:
        params['enable_early_init'] = enable_early_init

    return client.call('sock_impl_set_options', params)


def sock_set_default_impl(client, impl_name=None):
    """Set the default socket implementation.

    Args:
        impl_name: name of socket implementation, e.g. posix
    """
    params = {}

    params['impl_name'] = impl_name

    return client.call('sock_set_default_impl', params)
