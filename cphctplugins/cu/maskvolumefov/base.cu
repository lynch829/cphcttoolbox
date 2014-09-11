/*
#
# --- BEGIN_HEADER ---
#
# kernels - maskvolumefov CUDA Kernels
# Copyright (C) 2012-2013  The CT-Toolbox Project lead by Brian Vinter
#
# This file is part of CT-Toolbox.
#
# CT-Toolbox is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# CT-Toolbox is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
#
# -- END_HEADER ---
#
*/

/* Constants */

#define Z_SLICE_SIZE (rt_y_voxels*rt_x_voxels)

/* Flat indexing macros */

#define VOLUME_SLICE_IDX(y,x) ((y*rt_x_voxels)+x)

/* 
 * Mask reconstructed volume to specified FoV radius
 */

KERNEL void mask_volume_fov(
            GLOBALMEM float *recon_data,
            GLOBALMEM float *fov_mask) {
   
   unsigned int y = GET_GLOBAL_ID_Y;   
   unsigned int x = GET_GLOBAL_ID_X;

   unsigned int i;
   unsigned int voxel_idx = VOLUME_SLICE_IDX(y,x);

   float fov_voxel_mask = fov_mask[voxel_idx];

   for (i=0; i<rt_chunk_size; i++) {
      recon_data[voxel_idx] *= fov_voxel_mask;
      voxel_idx += Z_SLICE_SIZE;
   }
}
