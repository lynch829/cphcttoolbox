/*
#
# --- BEGIN_HEADER ---
#
# kernels - flux2proj CUDA Kernels
# Copyright (C) 2011  The CT-Toolbox Project lead by Brian Vinter
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

#define DETECTOR_SIZE (rt_detector_rows*rt_detector_columns)


/* Flat indexing macros */

#define PROJ_IDX(y,x) (y*rt_detector_columns+x)


/* 
 * Convert stacked intensity measurement data values from intensity/flux
 * to attenuation projections using logarithmic normalization with previously
 * prepared air_norm and zero_norm values. This is a quite common operation
 * to translate measured raw integer intensity values to actual projection
 * values.
 */

#ifdef plugin_rt_air_ref_pixel_idx
__global__ void flux2proj(float *proj_data,
			  float *proj_ref_pixel_vals,
			  unsigned int *proj_count,
			  float *zero_norm,
			  float *air_norm) {
#else
__global__ void flux2proj(float *proj_data,
			  unsigned int *proj_count,
			  float *zero_norm,
			  float *air_norm) {
#endif
   unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;   
   unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

   unsigned int i;
   unsigned int pixel_idx = PROJ_IDX(y,x);
   
   float zero_val = zero_norm[pixel_idx];
   float air_val = air_norm[pixel_idx];
   float pixel_val;
   
   
#ifdef plugin_rt_air_ref_pixel_idx
   unsigned int air_ref_idx = plugin_rt_air_ref_pixel_idx;
   float air_val_diff;
   float air_val_ref = air_norm[plugin_rt_air_ref_pixel_idx];
   float init_air_val = air_val;
#endif
   
   for (i=0; i<*proj_count; i++) {
      
#ifdef plugin_rt_air_ref_pixel_idx
      air_val_diff = air_val_ref - proj_ref_pixel_vals[i];
      pixel_val = logf(init_air_val - air_val_diff - zero_val) -
	          logf(proj_data[pixel_idx] - zero_val);
      
#else
      pixel_val = air_val - logf(proj_data[pixel_idx] - zero_val);
#endif 
      // In accordance to the IEEE-754R standard, 
      // if one of the input parameters to fminf(), fmin(), fmaxf(), or fmax() 
      // is NaN, but not the other, the result is the non-NaN parameter.

      // NaN's are set to zero
      pixel_val = fmaxf(pixel_val, 0.0f);
      
      // INF's are set to zerod air value
      pixel_val = fminf(pixel_val, air_val);
      
      proj_data[pixel_idx] = pixel_val;
      
#ifdef plugin_rt_air_ref_pixel_idx  
      air_ref_idx += DETECTOR_SIZE;
#endif
      pixel_idx += plugin_rt_proj_size;
   }
}
