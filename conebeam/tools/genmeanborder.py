#!/usr/bin/python
# -*- coding: utf-8 -*-

#
# --- BEGIN_HEADER ---
#
# genmeanborder - cone beam projection border manipulator
# Copyright (C) 2012  The Cph CT Toolbox Project lead by Brian Vinter
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

"""Generate mean value border projs from cone beam projection files"""

import sys

from cphct.cone.conf import float_value, int_value, colon_int_values, \
    default_cone_npy_conf, default_cone_npy_opts, enable_conf_option, \
    engine_opts, engine_conf, parse_setup, ParseError
from cphct.cone.npy.io import fill_cone_npy_conf
from cphct.npycore import average, log, exp
from cphct.cone.npycore.utils import inline_tool_helper, default_tool_opts, \
    default_init_tool as init_tool, default_exit_tool as exit_tool, \
    engine_app_aliases

app_names = [__file__]


def tool_opts():
    """Options for this tool

    Returns
    -------
    output : dict
        Returns a dictionary of processed options helper dictionaries.
    """

    opts = default_tool_opts()
    opts.update({
        'meanborder_center': {
            'long': 'meanborder-center',
            'short': None,
            'args': str,
            'handler': colon_int_values,
            'default': None,
            'description': 'Average all pixels outside these center pixels ' \
                + '(row0:rowN:col0:colN)',
            },
        'meanborder_rate': {
            'long': 'meanborder-rate',
            'short': None,
            'args': str,
            'handler': int_value,
            'default': 2,
            'description': 'Number of pixels to average and pair outside ' \
                + 'center pixels',
            },
        'meanborder_linear': {
            'long': 'meanborder-linear',
            'short': None,
            'args': str,
            'handler': bool,
            'default': False,
            'description': 'Use linear or logarithmic mean - intensity or ' \
                + 'attenuation values',
            },
        })
    return opts

def tool_conf():
    """For enabling option of same name in conf

    Returns
    -------
    output : dict
        Returns a configuration dictionary where the related options are set.
    """

    return enable_conf_option(tool_opts())

def core_tool(projs, conf, tool_state):
    """Core tool function: takes a chunk of projections and returns
    the same processed chunk based on conf and state settings.

    Parameters
    ----------
    projs : array
        An array of stacked projections to process.
    conf : dict
        A dictionary of configuration options.
    tool_state : dict
        A dictionary of tool helper variables.
    """

    # Find area to keep

    (first_row, last_row, _, _) = conf['collapseborder_center']
    end_row = last_row + 1
    meanborder_rate = conf['meanborder_rate']
    (total_projs, height, width) = projs.shape
    for index in xrange(total_projs):

        proj = projs[index]

        # Broadcast multi row average for rows outside center
        # If meanborder_rate is 2, rows are averaged pairwise to same values
        # halving the effective resolution for the border rows.

        for row in range(0, first_row + 1, meanborder_rate):
            if row + meanborder_rate > first_row:
                continue
            view = projs[index, row:row + meanborder_rate, :]
            if conf['meanborder_linear']:
                mean = average(proj[row:row + meanborder_rate, :],
                               axis=0)
            else:
                mean = -log(average(exp(-proj[row:row
                            + meanborder_rate, :]), axis=0))
            view[:] = mean
        view = projs[index, first_row:end_row, :]
        mean = proj[first_row:end_row]
        view[:] = mean
        for row in range(height, end_row, -meanborder_rate):
            if row - meanborder_rate < end_row:
                continue
            view = projs[index, row - meanborder_rate:row, :]
            if conf['meanborder_linear']:
                mean = average(proj[row - meanborder_rate:row, :],
                               axis=0)
            else:
                mean = -log(average(exp(-proj[row - meanborder_rate:
                            row, :]), axis=0))
            view[:] = mean
    return projs


def main(conf):
    """Generate mean value border projs

    Parameters
    ----------
    conf : dict
        A dictionary of configuration options.
    """

    fill_cone_npy_conf(conf)
    inline_tool_helper(app_names, conf, init_tool, core_tool, exit_tool)


if __name__ == '__main__':
    cone_cfg = default_cone_npy_conf()

    # Override default value for preprocessed settings and engine

    cone_cfg.update(tool_conf())
    cone_cfg.update(engine_conf())

    cone_opts = default_cone_npy_opts()

    # Override default no-op action for preprocessed settings and engine

    cone_opts.update(tool_opts())
    cone_opts.update(engine_opts())
    try:
        cone_cfg = parse_setup(sys.argv, app_names, cone_opts, cone_cfg)
    except ParseError, err:
        print 'ERROR: %s' % err
        sys.exit(1)
    app_names += engine_app_aliases(app_names, cone_cfg)
    main(cone_cfg)
