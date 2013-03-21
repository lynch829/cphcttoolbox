#!/usr/bin/python
# -*- coding: utf-8 -*-

#
# --- BEGIN_HEADER ---
#
# centerfdk - center slice fdk reconstruction wrapper
# Copyright (C) 2011  The Cph CT Toolbox Project lead by Brian Vinter
#
# This file is part of Cph CT Toolbox.
#
# Cph CT Toolbox is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# Cph CT Toolbox is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301,
# USA.
#
# -- END_HEADER ---
#

"""Cph CT Toolbox center slice FDK implementation wrapper to merge all
backends"""

import sys

from cphct.fan.centerfdk.conf import parse_setup, engine_opts, \
    engine_conf, default_centerfdk_npy_opts, default_centerfdk_cu_opts, \
    default_centerfdk_npy_conf, default_centerfdk_cu_conf, ParseError

app_names = ['centerfdk']

if __name__ == '__main__':

    # Use two steps: parse just engine before specific parse and main

    base_cfg = {}
    base_cfg.update(default_centerfdk_npy_conf())
    base_cfg.update(default_centerfdk_cu_conf())

    # Override default value for engine

    base_cfg.update(engine_conf())
    base_opts = {}
    base_opts.update(default_centerfdk_npy_opts())
    base_opts.update(default_centerfdk_cu_opts())

    # Override default no-op action for engine

    base_opts.update(engine_opts())
    try:
        base_cfg = parse_setup(sys.argv, app_names, base_opts, base_cfg)
    except ParseError, err:
        print 'ERROR: %s' % err
        sys.exit(1)
    engine = base_cfg['engine']
    if engine == 'numpy':
        from npycenterfdk import main
        cfg = default_centerfdk_npy_conf()
        opts = default_centerfdk_npy_opts()
        app_names.append('npycenterfdk')
    elif engine == 'cuda':
        from cucenterfdk import main
        cfg = default_centerfdk_cu_conf()
        opts = default_centerfdk_cu_opts()
        app_names.append('cucenterfdk')
    else:
        print 'Unknown engine: %s' % engine
        sys.exit(1)
    try:
        cfg = parse_setup(sys.argv, app_names, opts, cfg)
    except ParseError, err:
        print 'ERROR: %s' % err
        sys.exit(1)
    main(cfg, opts)
