/*
#
# --- BEGIN_HEADER ---
#
# kernels - flux2proj OpenCL Kernels
# Copyright (C) 2011-2013  The CT-Toolbox Project lead by Brian Vinter
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

/* Macros fixing OpenCL quirks */

/* 
   OpenCL does not define Xf() fuctions like C99 and CUDA does, but implicitly
   applies the same function by means of overloading:
   Simply replace Xf(...) by X(...) in the code.
*/

#define logf(x) log(x)
#define fminf(x, y) fmin(x, y)
#define fmaxf(x, y) fmax(x, y)

/* Flat indexing macros */

#define PROJ_IDX(y,x) (y*rt_detector_columns+x)


/* 
 * Convert stacked intensity measurement data values from intensity/flux
 * to attenuation projections using logarithmic normalization with previously
 * prepared air_norm and zero_norm values. This is a quite common operation
 * to translate measured raw integer intensity values to actual projection
 * values.
 */

KERNEL void flux2proj(
            GLOBALMEM float *proj_data,		      
            GLOBALMEM float *zero_norm,
		      GLOBALMEM float *air_norm,
            const unsigned int first_proj,
            const unsigned int last_proj
#ifdef plugin_rt_air_ref_pixel_flat_idx
            , GLOBALMEM float *proj_ref_pixel_vals) {
#else
         ) {
#endif
   unsigned int y = GET_GLOBAL_ID_Y;   
   unsigned int x = GET_GLOBAL_ID_X;

   unsigned int i;
   unsigned int pixel_idx = PROJ_IDX(y,x);
   
   float zero_val = zero_norm[pixel_idx];
   float air_val = air_norm[pixel_idx];
   float pixel_val;
   
   
#ifdef plugin_rt_air_ref_pixel_flat_idx
   float air_val_diff;
   float air_val_ref = air_norm[plugin_rt_air_ref_pixel_flat_idx];
   float zero_val_ref = zero_norm[plugin_rt_air_ref_pixel_flat_idx];
   float init_air_val = air_val;
#endif
   
   // NOTE: 'air' values are dark current corrected in base.py

   for (i=first_proj; i<=last_proj; i++) {
      
#ifdef plugin_rt_air_ref_pixel_flat_idx
      air_val_diff = air_val_ref - (proj_ref_pixel_vals[i] - zero_val_ref);
      air_val = logf(init_air_val - air_val_diff);
      pixel_val = air_val - logf(proj_data[pixel_idx] - zero_val);
      
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

      pixel_idx += plugin_rt_proj_size;
   }
}
