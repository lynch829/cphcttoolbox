#!/usr/bin/python
# -*- coding: utf-8 -*-

#
# --- BEGIN_HEADER ---
#
# misc - numpy core misc helpers
# Copyright (C) 2011-2012  The Cph CT Toolbox Project lead by Brian Vinter
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

"""Numpy core misc helper functions"""

from cphct.npycore import ones, linspace, atleast_1d, broadcast_arrays


def size_from_shape(shape):
    """Calculate size of array with given shape

    Parameters
    ----------
    shape : tuple of int
        Shape of some matrix.

    Returns
    -------
    output : int
        Returns the number of elements in a matrix with given shape. That is,
        the product of the matrix dimensions.
    """

    size = 1
    for val in shape:
        size *= val
    return size


def linear_coordinates(
    coord_min,
    coord_max,
    coord_count,
    coord_bounds,
    fdt,
    ):
    """
    Generates an array of *coord_count* linear coordinates from *coord_min* to
    *coord_max*. E.g. coordinates for each voxel/slice in the reconstruction.
    Please note that the special case where *coord_count* is 1 results in
    a single center coordinate with the average value of *coord_max* and
    *coord_min* which differs from from the linspace output of the single
    *coord_min* value.
    The *coord_bounds* flag is used to indicate that the *coord_min* and
    *coord_max* values are left and right boundaries and that interval center
    coordinate values are wanted. That is, the coordinate range is shrunk to a
    range half an interval length (e.g. voxel) shorter in each end.
    Examples:
    >>> linear_coordinates(-6.0, 6.0, 1, float32)
    array([ 0.], dtype=float32)
    >>> linear_coordinates(-6.0, 6.0, 1, True, float32)
    array([ 0.], dtype=float32)
    >>> linear_coordinates(-6.0, 6.0, 2, float32)
    array([-6.,  6.], dtype=float32)
    >>> linear_coordinates(-6.0, 6.0, 2, True, float32)
    array([-3.,  3.], dtype=float32)
    >>> linear_coordinates(-6.0, 6.0, 3, float32)
    array([-6.,  0.,  6.], dtype=float32)
    >>> linear_coordinates(-6.0, 6.0, 3, True, float32)
    array([-4.,  0.,  4.], dtype=float32)
    >>> linear_coordinates(-6.0, 6.0, 4, float32)
    array([-6., -2.,  2.,  6.], dtype=float32)
    >>> linear_coordinates(-6.0, 6.0, 4, True, float32)
    array([-4.5, -1.5,  1.5,  4.5], dtype=float32)
    
    Parameters
    ----------
    coord_min: float
       Minimum coordinate in array.
    coord_max: float
        Maximum coordinate in array.
    coord_count: int
       Number of coordinate values in array.
    coord_bounds: bool
       If min/max coordinates are outer boundaries and centers are wanted.
    fdt : dtype
        Output array data type.

    Returns
    -------
    output : ndarray
        Returns an array of *coord_count* linear coordinates in
        [*coord_min*:*coord_max*] with dtype *fdt*.
    """

    if coord_count == 1:
        coord_min = coord_max = 0.5 * (coord_max + coord_min)
    elif coord_bounds:
        center_off = 0.5 * ((coord_max - coord_min) / coord_count)
        coord_min, coord_max = coord_min + center_off, coord_max - center_off
    return fdt(linspace(coord_min, coord_max, coord_count, endpoint=True))
