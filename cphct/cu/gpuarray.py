#!/usr/bin/python
# -*- coding: utf-8 -*-

#
# --- BEGIN_HEADER ---
#
# gpuarray - pycuda.gpuarray replacement
# Copyright (C) 2011-2013  The Cph CT Toolbox Project lead by Brian Vinter
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

"""gpuarray - pycuda.gpuarray replacement"""

from cphct.npycore import integer
from pycuda import VERSION as PYCUDA_VERSION
from pycuda.gpuarray import *

def __getitem__(self, index):
    """
    From PyCUDA 2013.1.1 source:
    http://pypi.python.org/packages/source/p/pycuda/pycuda-2013.1.1.tar.gz
    """

    if not isinstance(index, tuple):
        index = (index, )

    new_shape = []
    new_offset = 0
    new_strides = []

    seen_ellipsis = False

    index_axis = 0
    array_axis = 0
    while index_axis < len(index):
        index_entry = index[index_axis]

        if array_axis > len(self.shape):
            raise IndexError('too many axes in index')

        if isinstance(index_entry, slice):
            (start, stop, idx_stride) = \
                index_entry.indices(self.shape[array_axis])

            array_stride = self.strides[array_axis]

            new_shape.append((stop - start) // idx_stride)
            new_strides.append(idx_stride * array_stride)
            new_offset += array_stride * start

            index_axis += 1
            array_axis += 1
        elif isinstance(index_entry, (int, integer)):

            array_shape = self.shape[array_axis]
            if index_entry < 0:
                index_entry += array_shape

            if not 0 <= index_entry < array_shape:
                raise IndexError('subindex in axis %d out of range'
                                 % index_axis)

            new_offset += self.strides[array_axis] * index_entry

            index_axis += 1
            array_axis += 1
        elif index_entry is Ellipsis:

            index_axis += 1

            remaining_index_count = len(index) - index_axis
            new_array_axis = len(self.shape) - remaining_index_count
            if new_array_axis < array_axis:
                raise IndexError('invalid use of ellipsis in index')
            while array_axis < new_array_axis:
                new_shape.append(self.shape[array_axis])
                new_strides.append(self.strides[array_axis])
                array_axis += 1

            if seen_ellipsis:
                raise IndexError('more than one ellipsis not allowed in index'
                                 )
            seen_ellipsis = True
        else:

            raise IndexError('invalid subindex in axis %d' % index_axis)

    while array_axis < len(self.shape):
        new_shape.append(self.shape[array_axis])
        new_strides.append(self.strides[array_axis])

        array_axis += 1

    return GPUArray(
        shape=tuple(new_shape),
        dtype=self.dtype,
        allocator=self.allocator,
        base=self,
        gpudata=int(self.gpudata) + new_offset,
        strides=tuple(new_strides),
        )


# }}}

if PYCUDA_VERSION < (2013, 1, 1):

    # If PyCUDA version is below 2013.1.1.
    # Inject __getitem__ from PyCUDA 2013.1.1 to support
    # multi-d GPUArray slicing

    GPUArray.__getitem__ = __getitem__
