#!/usr/bin/python
# -*- coding: utf-8 -*-

#
# --- BEGIN_HEADER ---
#
# base - cone beam back end functions shared by plugin and tools
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

"""Cph CT Toolbox cone beam back end functions shared by plugins and tools.
We try to separate I/O from the actual handlers so that they can be used
inside apps and in separate tools scripts.
"""

from cphct.log import logging
from cphct.npycore import zeros, arange
from cphct.npycore.misc import linear_coordinates
from cphct.npycore.utils import prepare_output


def extract_sinograms(projs, conf, out=None):
    """Extract sinograms from projs

    Parameters
    ----------
    projs : ndarray
        Array with projections to extract sinograms from.
    conf : dict
        Configuration dictionary.
    out : ndarray, optional
        Output argument. 
        This must have the exact kind that would be returned 
        if it was not used.
        In particular, it must have the right type, must be C-contiguous,
        and its dtype must be the dtype that would be returned 
        without *out* set.
        This is a performance feature. 
        Therefore, if these conditions are not met,
        an exception is raised,  instead of attempting to be flexible.

    Returns
    -------
    output : ndarray
        Stacked sinograms in ndarray.
    """

    (total_projs, height, width) = projs.shape

    if out is None:
        out = prepare_output((height, total_projs, width), conf)

    for index in range(total_projs):
        proj = projs[index]
        for row in range(height):
            out[row, index, :] = proj[row, :]
    return out


def extract_slices(projs, conf, out=None):
    """Extract slices from projs

    Parameters
    ----------
    projs : ndarray
        Array with projections to extract slices from.
    conf : dict
        Configuration dictionary.
    out : ndarray, optional
        Output argument. 
        This must have the exact kind that would be returned 
        if it was not used.
        In particular, it must have the right type, must be C-contiguous,
        and its dtype must be the dtype that would be returned 
        without *out* set.
        This is a performance feature. 
        Therefore, if these conditions are not met,
        an exception is raised,  instead of attempting to be flexible.

        
    Returns
    -------
    output : ndarray
        Stacked slices in ndarray.
    """

    fdt = conf['data_type']
    odt = conf['output_data_type']
    (total_projs, height, width) = projs.shape

    # Handle implicit ends

    if conf['last_row'] < 0:
        conf['last_row'] = height - 1
    if conf['last_column'] < 0:
        conf['last_column'] = width - 1

    # Avoid interpolation only due to limited precision

    first_row = fdt(conf['first_row']).round(6)
    last_row = fdt(conf['last_row']).round(6)
    first_column = fdt(conf['first_column']).round(6)
    last_column = fdt(conf['last_column']).round(6)
    rows = int((1 + last_row - first_row).round())
    cols = int((1 + last_column - first_column).round())
    row_indices = arange(rows, dtype=fdt) + fdt(first_row)

    if out is None:
        out = prepare_output((total_projs, rows, cols), conf)

    for index in range(total_projs):
        proj = projs[index]
        col_view = proj[:, first_column:last_column + 1]

        # For fractional rows we interpolate linearly before and after row

        for row_index in row_indices:
            row_offset = int((row_index - first_row).round())
            logging.debug('handling row %s (%s)' % (row_index,
                          row_offset))
            if row_index.round() == row_index:
                out[index, row_offset] = col_view[row_index]
            else:
                before_row = int(row_index)
                after_row = before_row + 1
                frac = row_index - before_row
                logging.debug('interpolating rows %s and %s (%f)'
                              % (before_row, after_row, frac))
                interpolated = (1 - frac) * col_view[before_row] + frac \
                    * col_view[after_row]

                # Avoid auto-cast to (u)int rounding fractions >= 0.5 to zero

                if odt(0.9) == 0:
                    interpolated.round()
                out[index, row_offset] = interpolated
                logging.debug('interpolated rows %d:%d with sum %f'
                              % (before_row, after_row, out[index,
                              row_offset].sum()))
    return out


def gap_offset_helper(gap_list, pixel_offsets):
    """Parse a list of accumulated pixel gap tuples and fill pixel_offsets
    with actual pixel center offsets. Please note that gaps are accumulated
    around the detector center. 

    Parameters
    ----------
    gap_list : list of tuples
        List of gap positions in pixels
    pixel_offsets : ndarray
        Array to fill with offsets

    Returns
    -------
    output : ndarray
        Same pixel_offsets array filled with individual pixel offsets.
    """

    pixels = len(pixel_offsets)
    gap_list.sort()
    (neg_gaps, pos_gaps) = ([], [])
    for (start, end) in gap_list:
        if start < 0:
            neg_gaps.append((start, min(end, 0)))
        if end > 0:
            pos_gaps.append((max(0, start), end))

    neg_gaps.reverse()
    logging.debug('neg_gaps %s, pos_gaps %s' % (neg_gaps, pos_gaps))

    neg_offsets = pixel_offsets[:pixels / 2][::-1]
    pos_offsets = pixel_offsets[pixels / 2:]
    offset = 0.0
    for (start, end) in pos_gaps:
        first = int(start - offset)
        cur_offset = end - start
        offset += cur_offset
        logging.debug('pos start %f end %f first %d %f' % (start, end,
                      first, offset))
        pos_offsets[first:] += cur_offset

    offset = 0.0
    for (start, end) in neg_gaps:
        first = -int(end - offset)
        cur_offset = end - start
        offset -= cur_offset
        logging.debug('neg start %f end %f first %d %f' % (start, end,
                      first, offset))
        neg_offsets[first:] -= cur_offset

    return pixel_offsets


def resample_gapless(
    projs,
    conf,
    detector_row_gaps=None,
    detector_column_gaps=None,
    detector_resample_rows=-1,
    detector_resample_columns=-1,
    detector_gap_interpolation='linear',
    out=None,
    ):
    """Resample projs without gaps:
    Take gap locations measured in pixels from conf and calculate real pixel
    offset locations again measured in pixels. Resample to the same or a
    configured number of new gapless pixels filling out the original total
    detector size. Each resampled pixel center is calculated by distributing
    them linearly on the total width. For each of these locations the
    resampled pixel value is found using either linear (default) or
    nearest-neighbor interpolation along both detector axes.

    Parameters
    ----------
    projs : ndarray
        Array with projections to resample projections from.
    conf : dict
        Configuration dictionary.
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
    out : ndarray, optional
        Output argument. 
        This must have the exact kind that would be returned 
        if it was not used.
        In particular, it must have the right type, must be C-contiguous,
        and its dtype must be the dtype that would be returned 
        without *out* set.
        This is a performance feature. 
        Therefore, if these conditions are not met,
        an exception is raised,  instead of attempting to be flexible.
        
    Returns
    -------
    output : ndarray
        Stacked resampled projections in ndarray.
    """

    idt = conf['input_data_type']
    fdt = conf['data_type']
    (total_projs, height, width) = projs.shape
    resample_width = detector_resample_columns
    resample_height = detector_resample_rows
    if resample_width < 1:
        resample_width = width
    if resample_height < 1:
        resample_height = height

    if out is None:
        out = prepare_output((total_projs, resample_height,
                             resample_width), conf)

    u_gaps = detector_column_gaps
    v_gaps = detector_row_gaps

    logging.debug('parsed u_gaps: %s' % u_gaps)
    logging.debug('parsed v_gaps: %s' % v_gaps)

    raw_u_centers = linear_coordinates(-width / 2, width / 2, width,
            True, fdt)
    raw_v_centers = linear_coordinates(-height / 2, height / 2, height,
            True, fdt)
    logging.debug('raw_u_centers: %s' % raw_u_centers)
    logging.debug('raw_v_centers: %s' % raw_v_centers)
    u_offsets = zeros(width, dtype=fdt)
    v_offsets = zeros(height, dtype=fdt)
    gappy_u_centers = zeros(width, dtype=fdt)
    gappy_v_centers = zeros(height, dtype=fdt)

    gap_offset_helper(u_gaps, u_offsets)
    gap_offset_helper(v_gaps, v_offsets)

    logging.debug('created u_offsets: %s' % u_offsets)
    logging.debug('created v_offsets: %s' % v_offsets)
    gappy_u_centers = raw_u_centers + u_offsets
    gappy_v_centers = raw_v_centers + v_offsets

    logging.debug('created gappy_u_centers: %s' % gappy_u_centers)
    logging.debug('created gappy_v_centers: %s' % gappy_v_centers)
    detector_gappy_width = 2.0 * gappy_u_centers[-1] + 1
    detector_gappy_height = 2.0 * gappy_v_centers[-1] + 1
    resample_u_factor = detector_gappy_width / resample_width
    resample_v_factor = detector_gappy_height / resample_height
    resampled_u_centers = resample_u_factor \
        * linear_coordinates(-resample_width / 2, resample_width / 2,
                             resample_width, True, fdt)
    resampled_v_centers = resample_v_factor \
        * linear_coordinates(-resample_height / 2, resample_height / 2,
                             resample_height, True, fdt)
    logging.info('resampling to gapless %dx%d projections of %f cm x %f cm'
                  % (resample_height, resample_width,
                 conf['detector_pixel_height'] * resample_v_factor,
                 conf['detector_pixel_width'] * resample_u_factor))
    logging.debug('''original pixel centers in pixels:
%s
%s'''
                  % (gappy_u_centers, gappy_v_centers))
    logging.debug('''original pixel centers in cm:
%s
%s'''
                  % (gappy_u_centers * conf['detector_pixel_width'],
                  gappy_v_centers * conf['detector_pixel_height']))
    logging.debug('''resampled pixel centers in pixels:
%s
%s'''
                  % (resampled_u_centers, resampled_v_centers))

    # Resampled centers are in original pixel size coordinates

    logging.debug('''resampled pixel centers in cm:
%s
%s'''
                  % (resampled_u_centers * conf['detector_pixel_width'
                  ], resampled_v_centers * conf['detector_pixel_height'
                  ]))
    logging.debug('created resampled_u_centers: %s'
                  % resampled_u_centers)
    logging.debug('created resampled_v_centers: %s'
                  % resampled_v_centers)

    # Helpers for interpolation

    (below_index, above_index, left_index, right_index) = ([], [], [],
            [])

    # Find nearest neighbor in original pixels for all resample positions

    for row in xrange(resample_height):
        index = ([height - 1] + [i for i in xrange(height)
                 if gappy_v_centers[i] <= resampled_v_centers[row]])[-1]
        below_index.append(index)
        above_index.append(min(height - 1, index + 1))
    for col in xrange(resample_width):
        index = ([width - 1] + [i for i in xrange(width)
                 if gappy_u_centers[i] <= resampled_u_centers[col]])[-1]
        left_index.append(index)
        right_index.append(min(width - 1, index + 1))
    logging.debug('created left_index: %s' % left_index)
    logging.debug('created right_index: %s' % right_index)
    logging.debug('created below_index: %s' % below_index)
    logging.debug('created above_index: %s' % above_index)

    # Precalculate matrix of interpolation factors for each pixel only once
    # and use them for all projections. The four factors are the distances to
    # the centers of the nearest neighbor pixels along both axes.

    interpol_nw = zeros((resample_height, resample_width), dtype=fdt)
    interpol_ne = zeros((resample_height, resample_width), dtype=fdt)
    interpol_sw = zeros((resample_height, resample_width), dtype=fdt)
    interpol_se = zeros((resample_height, resample_width), dtype=fdt)
    for row in xrange(resample_height):
        (below, above) = (above_index[row], below_index[row])
        for col in xrange(resample_width):
            (left, right) = (left_index[col], right_index[col])
            if left == right:
                u_frac = 0.0
            else:
                u_frac = abs((resampled_u_centers[col]
                             - gappy_u_centers[left])
                             / (gappy_u_centers[right]
                             - gappy_u_centers[left]))
            if below == above:
                v_frac = 0.0
            else:
                v_frac = abs((resampled_v_centers[row]
                             - gappy_v_centers[above])
                             / (gappy_v_centers[above]
                             - gappy_v_centers[below]))

            # linear interpolation factors by default (fractional)

            (u_frac, v_frac) = (fdt(u_frac), fdt(v_frac))
            if detector_gap_interpolation == 'point':

                # nearest-neighbor interpolation by rouding to zero or one

                (u_frac, v_frac) = (int(u_frac.round()),
                                    int(v_frac.round()))
            interpol_nw[row, col] = (1 - u_frac) * (1 - v_frac)
            interpol_ne[row, col] = u_frac * (1 - v_frac)
            interpol_sw[row, col] = (1 - u_frac) * v_frac
            interpol_se[row, col] = u_frac * v_frac
    logging.debug('interpol helpers center: %f %f %f %f'
                  % (interpol_nw[resample_height / 2, resample_width
                  / 2], interpol_ne[resample_height / 2, resample_width
                  / 2], interpol_sw[resample_height / 2, resample_width
                  / 2], interpol_se[resample_height / 2, resample_width
                  / 2]))

    for index in xrange(total_projs):
        proj = projs[index]
        for row in xrange(resample_height):
            (below, above) = (above_index[row], below_index[row])
            for col in xrange(resample_width):
                (left, right) = (left_index[col], right_index[col])
                pixel = interpol_nw[row, col] * proj[above, left] \
                    + interpol_ne[row, col] * proj[above, right] \
                    + interpol_sw[row, col] * proj[below, left] \
                    + interpol_se[row, col] * proj[below, right]

                # Avoid auto-cast to (u)int rounding fractions >= 0.5 to zero

                if idt(0.9) == 0:
                    pixel.round()
                out[index, row, col] = pixel
                if row == resample_height / 2 and col == resample_width \
                    / 2:
                    logging.debug('interpol %d, %d, %d, %d for (%d, %d)'
                                   % (
                        left,
                        right,
                        above,
                        below,
                        row,
                        col,
                        ))
                    logging.debug('interpol %f %f %f %f %f %f to %f %f (%f)'
                                   % (
                        proj[above, left],
                        proj[below, left],
                        proj[above, right],
                        proj[below, right],
                        u_frac,
                        v_frac,
                        pixel,
                        out[index, row, col],
                        pixel - proj[0, 0],
                        ))

    return out


