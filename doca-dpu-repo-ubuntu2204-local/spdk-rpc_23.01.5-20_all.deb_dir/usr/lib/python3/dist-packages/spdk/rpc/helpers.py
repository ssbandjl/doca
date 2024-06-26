#  SPDX-License-Identifier: BSD-3-Clause
#  Copyright (C) 2019 Intel Corporation.
#  All rights reserved.

import sys

deprecated_aliases = {}


def deprecated_alias(old_name):
    def wrap(f):
        def old_f(*args, **kwargs):
            ret = f(*args, **kwargs)
            sys.stderr.write("{} is deprecated, use {} instead.\n".format(
                old_name, f.__name__))
            return ret
        old_f.__name__ = old_name
        deprecated_aliases[old_name] = f.__name__
        setattr(sys.modules[f.__module__], old_name, old_f)
        return f
    return wrap
