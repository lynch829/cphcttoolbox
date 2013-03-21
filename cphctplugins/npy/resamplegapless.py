#!/usr/bin/python
# -*- coding: utf-8 -*-

#
# --- BEGIN_HEADER ---
#
# resamplegapless - resample plugin to remove known projection pixel gaps
# Copyright (C) 2012-2013  The Cph CT Toolbox Project lead by Brian Vinter
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

"""Resample plugin to remove known pixel gaps from projections"""

from cphct.npycore import arange, zeros_like
from cphct.cone.npycore.utils import resample_gapless
from cphct.cone.conf import gap_pair_values
from cphct.conf import allowed_interpolation
from cphct.npycore.io import get_npy_data

# Internal plugin state for individual plugin instances

__plugin_state__ = {}


def plugin_init(
    conf,
    detector_row_gaps=None,
    detector_column_gaps=None,
    detector_resample_rows=-1,
    detector_resample_columns=-1,
    detector_gap_interpolation='linear',
    ):
    """Plugin init function called once with full configuration upon plugin
    load. Called before any hooks, so it may be useful for global
    preparations of the plugin.
    Any values for use in subsequent hooks can be saved in the plugin-specific
    __plugin_state__ dictionary.

    Just check args in this case.

    Parameters
    ----------
    conf : dict
        A dictionary of configuration options
    detector_row_gaps : list of (float, float)
        list of row gap start and end postions measured in detector pixels.
    detector_column_gaps : list of (float, float)
        list of column gap start and end positions measured in detector
        pixels.
    detector_resample_rows : int
        number of detector rows in resampled projection.
    detector_resample_columns : int
        number of detector columns in resampled projection.
    detector_gap_interpolation : str
        interpolation mode in resampling.

    Raises
    ------
    ValueError
        If provided row and column args are not valid lists of floating point
        tuples and integer values respectively.
    """

    __plugin_state__['name'] = __name__

    detector_resample_rows = int(detector_resample_rows)
    detector_resample_columns = int(detector_resample_columns)

    if not detector_row_gaps:
        detector_row_gaps = []
    else:
        detector_row_gaps = gap_pair_values(detector_row_gaps)
    __plugin_state__['detector_row_gaps'] = detector_row_gaps

    if not detector_column_gaps:
        detector_column_gaps = []
    else:
        detector_column_gaps = gap_pair_values(detector_column_gaps)

    __plugin_state__['detector_column_gaps'] = detector_column_gaps

    if detector_resample_rows == -1:
        __plugin_state__['detector_resample_rows'] = \
            conf['detector_rows']
    else:
        __plugin_state__['detector_resample_rows'] = \
            detector_resample_rows

    if detector_resample_columns == -1:
        __plugin_state__['detector_resample_columns'] = \
            conf['detector_columns']
    else:
        __plugin_state__['detector_resample_columns'] = \
            detector_resample_columns

    if not detector_gap_interpolation in allowed_interpolation:
        raise ValueError('invalid resamplegapless interpolation method')
    __plugin_state__['detector_gap_interpolation'] = \
        detector_gap_interpolation

    __plugin_state__['tmp_projs_data'] = zeros_like(get_npy_data(conf,
            'projs_data'))


def plugin_exit(
    conf,
    detector_row_gaps=None,
    detector_column_gaps=None,
    detector_resample_rows=-1,
    detector_resample_columns=-1,
    detector_gap_interpolation='linear',
    ):
    """Plugin exit function called once with full configuration at the end of
    execution. Called after all hooks are finished, so it may be useful
    for global clean up after the plugin.
    Any memory allocations that need to be released can be handled here.

    Nothing to do in this case.

    Parameters
    ----------
    conf : dict
        A dictionary of configuration options.
    detector_row_gaps : list of (float, float)
        list of row gap start and end postions measured in detector pixels.
    detector_column_gaps : list of (float, float)
        list of column gap start and end positions measured in detector
        pixels.
    detector_resample_rows : int
        number of detector rows in resampled projection.
    detector_resample_columns : int
        number of detector columns in resampled projection.
    detector_gap_interpolation : str
        interpolation mode in resampling.
    """

    __plugin_state__.clear()


def preprocess_input(
    input_data,
    input_meta,
    conf,
    detector_row_gaps=None,
    detector_column_gaps=None,
    detector_resample_rows=-1,
    detector_resample_columns=-1,
    detector_gap_interpolation='linear',
    ):
    """Resample input using args for gap and resampling information

    Parameters
    ----------
    input_data : ndarray
        array to process.
    input_meta : list of dict
        List of meta data dictionaries matching input_data.
    conf : dict
        A dictionary of configuration options.
    detector_row_gaps : list of (float, float)
        list of row gap start and end postions measured in detector pixels.
        detector_column_gaps : list of (float, float)
        list of column gap start and end positions measured in detector
        pixels.
    detector_resample_rows : int
        number of detector rows in resampled projection.
    detector_resample_columns : int
        number of detector columns in resampled projection.
    detector_gap_interpolation : str
        interpolation mode in resampling.

    Returns
    -------
    output : tuple of ndarray and list
        Returns a tuple of resampled array with the same number of
        projections but with detector_resample_rows x
        detector_resample_columns projections and meta list.
    """

    # Raise error if input is not a numpy array

    if not hasattr(input_data, 'dtype'):
        raise ValueError('invalid resamplegapless preprocess input array'
                         )

    detector_row_gaps = __plugin_state__['detector_row_gaps']
    detector_column_gaps = __plugin_state__['detector_column_gaps']
    detector_resample_rows = __plugin_state__['detector_resample_rows']
    detector_resample_columns = \
        __plugin_state__['detector_resample_columns']
    detector_gap_interpolation = \
        __plugin_state__['detector_gap_interpolation']
    tmp_projs_data = __plugin_state__['tmp_projs_data']

    resample_gapless(
        input_data,
        conf,
        detector_row_gaps,
        detector_column_gaps,
        detector_resample_rows,
        detector_resample_columns,
        detector_gap_interpolation,
        out=tmp_projs_data,
        )
    input_data[:] = tmp_projs_data

    return (input_data, input_meta)


if __name__ == '__main__':
    from cphct.npycore import allowed_data_types
    conf = {
        'detector_columns': 8,
        'detector_rows': 2,
        'total_projs': 2,
        'detector_pixel_width': 1,
        'detector_pixel_height': 1,
        'detector_row_gaps': [],
        'detector_column_gaps': [(-1.5, -0.5), (0.5, 1.5)],
        'detector_resample_rows': -1,
        'detector_resample_columns': -1,
        'detector_gap_interpolation': 'linear',
        }
    total_elems = conf['total_projs'] * conf['detector_columns'] \
        * conf['detector_rows']
    for type_name in allowed_data_types:
        conf['input_precision'] = type_name
        data_type = allowed_data_types[type_name]
        conf['precision'] = conf['output_precision'] = 'float32'
        data = arange(0, total_elems, dtype=data_type)
        data.shape = (conf['total_projs'], conf['detector_rows'],
                      conf['detector_columns'])
        print 'Gapless resample data %s (%s)' % (data, data.dtype.name)
        out = resample_gapless(
            data,
            conf,
            conf['detector_row_gaps'],
            conf['detector_column_gaps'],
            conf['detector_resample_rows'],
            conf['detector_resample_columns'],
            conf['detector_gap_interpolation'],
            )

        print 'Resampled to %s (%s)' % (out, out.dtype.name)
