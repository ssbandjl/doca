#  SPDX-License-Identifier: BSD-3-Clause
#  Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.
#  All rights reserved.

from spdk.rpc.helpers import deprecated_alias


def mlx5_scan_accel_module(client, qp_size=None, cq_size=None, num_requests=None, split_mb_blocks=None, allowed_devs=None,
                           siglast=None, merge=None, qp_per_domain=None):
    """Enable mlx5 accel module. Scans all mlx5 devices which can perform needed operations

    Args:
        qp_size: Qpair size. (optional)
        num_requests: size of a global requests pool per mlx5 device (optional)
        merge: merge tasks in the sequence when possible (optional)
    """
    params = {}

    if qp_size is not None:
        params['qp_size'] = qp_size
    if cq_size is not None:
        params['cq_size'] = cq_size
    if num_requests is not None:
        params['num_requests'] = num_requests
    if split_mb_blocks is not None:
        params['split_mb_blocks'] = split_mb_blocks
    if allowed_devs is not None:
        params['allowed_devs'] = allowed_devs
    if siglast is not None:
        params['siglast'] = siglast
    if merge is not None:
        params['merge'] = merge
    if qp_per_domain is not None:
        params['qp_per_domain'] = qp_per_domain
    return client.call('mlx5_scan_accel_module', params)


def accel_mlx5_dump_stats(client, level=None):

    params = {}

    if level is not None:
        params['level'] = level
    return client.call('accel_mlx5_dump_stats', params)
