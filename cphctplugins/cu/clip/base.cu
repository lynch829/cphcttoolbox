/*
#
# --- BEGIN_HEADER ---
#
# kernels - clip OpenCL Kernels
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
 * Clip output using range information
 */

KERNEL void clip(
            GLOBALMEM float *recon_data,
            float clip_min,
            float clip_max) {
   
   unsigned int y = GET_GLOBAL_ID_Y;   
   unsigned int x = GET_GLOBAL_ID_X;

   unsigned int i;
   unsigned int voxel_idx = VOLUME_SLICE_IDX(y,x);
   float pixel_val;

   for (i=0; i<rt_chunk_size; i++) {      
      pixel_val = recon_data[voxel_idx];
      pixel_val = (clip_min<pixel_val)?pixel_val:clip_min;
      pixel_val = (clip_max>pixel_val)?pixel_val:clip_max;
      recon_data[voxel_idx] = pixel_val;
      
      voxel_idx += Z_SLICE_SIZE;
   }
}
