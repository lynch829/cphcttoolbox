/*
#
# --- BEGIN_HEADER ---
#
# kernels - Katsevich Cone Beam Reconstruction CUDA Kernels
# Copyright (C) 2011-2014  The CT-Toolbox Project lead by Brian Vinter
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

/* For debugging */
/*
#include <stdio.h>
*/

/*
Pure kernels for spiral cone beam CT in CUDA using the Katsevich algorithm

Requires external init of geometry configuration like scan_radius and so on.
This may come from init.cu or be hard coded from runtime generated code.
*/


/* Constants */


/* 
   CUDA defines Xf() fuctions like C99, so just include header and use them 
   without further ado.
   Also gives us thing like CUDA_VERSION etc.
*/

#include <cuda.h>


/* Macros fixing CUDA quirks */

// Must force positive values in powf
#define pow2f(x) powf(abs(x), 2)


/* 
   Indices for differentiation during filtering.

   We use a naming of PROJ_D where PROJ refers to current or next projection
   and D describes the row and column offset in that projection:
   - cur_D thus refers to current projection with D starting in the south west
     (i.e. cur_sw). 
   - next_D refers to next projection with same system for D
   In the current projection the pixel in the row above is thus cur_nw.
*/

#define cur_sw 0
#define cur_se 1
#define cur_nw 2
#define cur_ne 3
#define next_sw 4
#define next_se 5 
#define next_nw 6
#define next_ne 7 

/* 
   TODO: switch prev/cur/next values to just to pointer moves?
   In principle it is better to only swap pointers between iterations, but if
   it results in the vars being stored in slow memory it may be more efficient
   to keep the arrays with constant offsets and let the compiler keep these
   constant offset array values in registers.
*/

/* For array triples used e.g in boundary weights */

#define prev 0
#define cur 1
#define next 2
#define cycle_next(array) array[prev] = array[cur]; array[cur] = array[next]
#define cycle_prev(array) array[next] = array[cur]; array[cur] = array[prev]


/* abstract project pixel and voxel offset */

#define local_elems (BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z)
#define proj_elems (rt_detector_rows * rt_detector_columns)
#define rebin_elems (rt_detector_rebin_rows * rt_detector_columns)
#define row_elems (rt_detector_columns)
#define chunk_elems (rt_y_voxels * rt_x_voxels)
#define y_elems (rt_x_voxels)
#define pixel_offset(proj, row, col) ((proj) * proj_elems + (row) * row_elems + (col))
#define rebin_offset(proj, rebin_row, col) ((proj) * rebin_elems + (rebin_row) * row_elems + (col))
#define voxel_offset(x, y, z) ((z) * chunk_elems + (y) * y_elems + (x))
#define local_offset(x, y, z) ((z) * BLOCK_DIM_Y * BLOCK_DIM_X + (y) * BLOCK_DIM_X + (x))


/* Thread assignment for all kernels */
/* please note that order in filtering block is reversed to col, row, proj */

/* filter called with (Bx, By, 1) x (Gx, Gy, Gz) 
   where Gz=projs, Bx*Gx=cols and By*Gy=rows */
/* rebin called with (Bx, By, 1) x (Gx, Gy, Gz)
   where Gz=projs, Bx*Gx=cols and By*Gy=rebin_rows */
/* backproj called with (Bx, By, 1) x (Gx, Gy, Gz) 
   where Gx=x_voxels, By*Gy=y_voxels and Bx*Gz=chunk_size */

// iterate fastest over col, then row, then proj

#define thread_filtering_row() (GET_GLOBAL_ID_Y)
#define thread_filtering_col() (GET_GLOBAL_ID_X)
#define thread_filtering_proj_local() (GET_GLOBAL_ID_Z) 
#define thread_filtering_rebin_row() (GET_GLOBAL_ID_Y)

// iterate fastest over x, then y, then z

#define thread_backproject_x() (GET_GLOBAL_ID_X)
#define thread_backproject_y() (GET_GLOBAL_ID_Y)
#define thread_backproject_z() (GET_GLOBAL_ID_Z)

/* Individual filtering steps for synchronized execution of each step. */

/*
  Differentiate individual pixels in chunk of projections. Each thread handles
  one pixel and blocks are launched to cover entire chunk in projections.
*/

KERNEL void flat_diff_chunk(const int first, 
			    const int last, 
			    GLOBALMEM float *gpu_input,
			    GLOBALMEM float *gpu_output) {

  /* please note that order in block is reversed to col, row, proj */

  int proj_local = thread_filtering_proj_local();
  int row = thread_filtering_row();
  int col = thread_filtering_col();
  int proj = proj_local + first;
  int offsets[9];

  float d_proj = -1.0f, d_row = -1.0f, d_col = -1.0f;
  float dia = -1.0f, dia_sqr = -1.0f, result = 0.0f;
  float row_coords, col_coords, row_sqr, col_sqr, row_col_prod;

  /* skip last index in each dimension to avoid index+1 out of bounds */

  if (proj >= last || 
      row >= rt_detector_rows - 1 || 
      col >= rt_detector_columns - 1)
    return;

  dia = rt_scan_diameter;
  dia_sqr = dia * dia;

  /* 
     Detector pixel center coordinates using coordinate system centered in
     the middle of the detector. 
     Please note that the row coordinate formula in (82) of the Noo paper        
     is wrong. It should be Nrows instead of Ncols and fig 4 indicates that      
     the center row coordinates are used which means it should include the       
     '-1' too to get the half pixel offset like for the curved detector.         
  */

  row_coords = rt_detector_pixel_height * 
      (float(row) + rt_detector_row_offset - 
      0.5f * float(rt_detector_rows - 1));

  col_coords = rt_detector_pixel_span * 
      (float(col) + rt_detector_column_offset -
      0.5f * float(rt_detector_columns - 1));

  row_sqr = row_coords * row_coords;
  col_sqr = col_coords * col_coords;
  row_col_prod = row_coords * col_coords;

  /* Differentiate using the chain rule and manual interpolation */
  /* extract neighbor pixel values using the naming scheme described above */

  offsets[cur_sw] = pixel_offset(proj_local, row, col);
  offsets[cur_se] = pixel_offset(proj_local, row, col+1);
  offsets[cur_nw] = pixel_offset(proj_local, row+1, col);
  offsets[cur_ne] = pixel_offset(proj_local, row+1, col+1);
  offsets[next_sw] = pixel_offset(proj_local+1, row, col);
  offsets[next_se] = pixel_offset(proj_local+1, row, col+1);
  offsets[next_nw] = pixel_offset(proj_local+1, row+1, col);
  offsets[next_ne] = pixel_offset(proj_local+1, row+1, col+1);

  d_proj = (gpu_input[offsets[next_sw]] - gpu_input[offsets[cur_sw]] +	
      gpu_input[offsets[next_nw]] - gpu_input[offsets[cur_nw]] +	
      gpu_input[offsets[next_se]] - gpu_input[offsets[cur_se]] +	
      gpu_input[offsets[next_ne]] - gpu_input[offsets[cur_ne]]) / 
      (4.0f * rt_delta_s);

  d_row = (gpu_input[offsets[cur_nw]] - gpu_input[offsets[cur_sw]] +	
      gpu_input[offsets[cur_ne]] - gpu_input[offsets[cur_se]] +	
      gpu_input[offsets[next_nw]] - gpu_input[offsets[next_sw]] +	
      gpu_input[offsets[next_ne]] - gpu_input[offsets[next_se]]) / 
      (4.0f * rt_detector_pixel_height);

  d_col = (gpu_input[offsets[cur_se]] - gpu_input[offsets[cur_sw]] +	
      gpu_input[offsets[cur_ne]] - gpu_input[offsets[cur_nw]] +	
      gpu_input[offsets[next_se]] - gpu_input[offsets[next_sw]] +	
      gpu_input[offsets[next_ne]] - gpu_input[offsets[next_nw]]) / 
      (4.0f * rt_detector_pixel_span);

  result = d_proj + d_col * (col_sqr + dia_sqr) / dia + 
      d_row * row_col_prod / dia;

  result *= dia / sqrt(col_sqr + dia_sqr + row_sqr);

  /* out of bounds check */
  /*
  if (offsets[cur_sw] >= (last-first)*rt_detector_rows*rt_detector_columns) {
    printf("ERROR: diff_chunk out of bounds! %d\n", offsets[cur_sw]);
    return;
  }
  */

  gpu_output[offsets[cur_sw]] = result;
}

/* 
   Forward rebin individual pixels in chunk of projections. Each thread handles
   one pixel and blocks are launched to cover entire chunk in rebinned
   projections.
 */

KERNEL void flat_fwd_rebin_chunk(const int first, 
				 const int last, 
				 GLOBALMEM float *gpu_input,
				 GLOBALMEM float *gpu_output) {

  /* please note that order in block is reversed to col, rebin_row, proj */

  int proj_local = thread_filtering_proj_local();
  int rebin_row = thread_filtering_rebin_row();
  int col = thread_filtering_col();
  int proj = proj_local + first;

  int row = -1, offset = -1;
  float fwd_rebin_row = -1.0f, row_scaled = -1.0f, row_frac = -1.0f;
  float rebin_scale = -1.0f, rebin_coord = -1.0f, col_coord = -1.0f;
  float result = 0.0f;

  /* skip last index in projs and cols */

  if (proj >= last || 
      rebin_row > rt_detector_rebin_rows || 
      col >= rt_detector_columns)
    return;

  rebin_scale = 2.0f * rt_progress_per_radian;

  rebin_coord = (-rt_pi / 2.0f) - rt_half_fan_angle +
      rt_detector_rebin_rows_height * rebin_row;

  col_coord = rt_detector_pixel_span * 
      (float(col) + rt_detector_column_offset - 
      0.5f * float(rt_detector_columns - 1));

  fwd_rebin_row = rebin_scale *
      (rebin_coord + (rebin_coord / tanf(rebin_coord)) *
      (col_coord / rt_scan_diameter));

  /* sign is inverted for shift and thus also for offset */

  row_scaled = fwd_rebin_row / rt_detector_pixel_height + 
      0.5f * float(rt_detector_rows) - rt_detector_row_offset;

  /* make sure row and row+1 are in valid row range */

  row_scaled = min(max(0.0f, row_scaled), 
      float(rt_detector_rows) - 2.0f);

  row = (int)row_scaled;
  row_frac = row_scaled - float(row);

  /* Interpolate the two row neighbors */

  offset = pixel_offset(proj_local, row, col);

  result = (1.0f - row_frac) * gpu_input[offset];

  offset = pixel_offset(proj_local, row+1, col);
  result += row_frac * gpu_input[offset];

  offset = rebin_offset(proj_local, rebin_row, col);

  /* out of bounds check */
  /*
  if (offset >= (last-first)*rt_detector_rebin_rows*rt_detector_columns) {
    printf("ERROR: fwd_rebin_chunk out of bounds! %d\n", offset);
    return;
  }
  */

  gpu_output[offset] = result;
}

/* 
   Convolution of individual pixels in chunk of rebinned projections. Each
   thread handles one pixel and blocks are launched to cover entire chunk in 
   rebinned projections.
 */

KERNEL void flat_convolve_chunk(const int first,
				const int last, 
				GLOBALMEM float *gpu_input,
				GLOBALMEM float *hilbert_ideal,
				GLOBALMEM float *gpu_output) {

  /* please note that order in block is reversed to col, rebin_row, proj */

  int proj_local = thread_filtering_proj_local();
  int rebin_row = thread_filtering_rebin_row();
  int col = thread_filtering_col();

  int offset = -1, conv_col = -1, i = -1;
  float result = 0.0f;

  SHAREDMEM float shared_hilbert_ideal[rt_kernel_width];

  /* Prefetch hilbert ideal to SHAREDMEM */

  offset = local_offset(GET_LOCAL_ID_X, GET_LOCAL_ID_Y, GET_LOCAL_ID_Z);
  while (offset < rt_kernel_width) {     
    shared_hilbert_ideal[offset] = hilbert_ideal[offset];
    offset += local_elems;
  }
  
  sync_shared_mem();

  /*
    TODO: use rectangular hilbert window as suggested in Noo paper?
  */
  /* 
     convolve(f, g)[n] is defined to be the sum of f[m]*g[n-m] for all m
     in range (-inf, inf).
     We only need the central slice of the resulting N+M-1 discrete 
     elements.
  */

  for (i = -rt_kernel_radius; i <= rt_kernel_radius; i++) {

    /* We reverse sum order to step through rows in a cache efficient way */

    conv_col = col + i;

    /* make sure conv_col index stays in rebin_row */

    conv_col = min(max(conv_col, 0), rt_detector_columns - 1);
    offset = rebin_offset(proj_local, rebin_row, conv_col);

    /* reversed hilbert access order to match reversed col order above */

    result += shared_hilbert_ideal[rt_kernel_radius - i] * gpu_input[offset];
  }

  offset = rebin_offset(proj_local, rebin_row, col);
  gpu_output[offset] = result;
}

/* 
   Reverse rebin individual pixels in chunk of projections. Each thread handles
   one pixel and blocks are launched to cover entire chunk in projections.
*/

KERNEL void flat_rev_rebin_chunk(const int first,
				 const int last,
				 GLOBALMEM float *gpu_input,
				 GLOBALMEM float *gpu_output) {

  /* please note that order in block is reversed to col, row, proj */

  int proj_local = thread_filtering_proj_local();
  int row = thread_filtering_row();
  int col = thread_filtering_col();

  int rebin_row = -1, rebin_col = -1, rebin = -1, offset = -1;

  /* We allocate three rebin elements to use easy-read prev, cur, next */

  float fracs[2], fwd_rebin_row[3], rebin_coords[3];
  float row_coord = -1.0f, col_coord = -1.0f, rebin_scale = -1.0f;
  float result = 0.0f;

  row_coord = rt_detector_pixel_height * 
      (float(row) + rt_detector_row_offset - 
      0.5f * float(rt_detector_rows - 1));

  col_coord = rt_detector_pixel_span * 
      (float(col) + rt_detector_column_offset - 
      0.5f * float(rt_detector_columns - 1));

  rebin_scale = 2.0f * rt_progress_per_radian;

  /* col offset ruins positive/negative half split */

  if (col_coord >= 0.0f) {

    /* column coordinate in positive range */

    rebin_row = 0;
    rebin_col = col;
    fracs[0] = 0.0f; fracs[1] = 1.0f;

    /* unrolled warm up round for loop - prepares cur values for first loop */

    rebin = -1;
    rebin_coords[next] = (-rt_pi / 2.0f) - rt_half_fan_angle + 
        rt_detector_rebin_rows_height * float(rebin + 1);

    fwd_rebin_row[next] = rebin_scale * (rebin_coords[next] +		
        (rebin_coords[next] / tanf(rebin_coords[next]))	*
        (col_coord / rt_scan_diameter));

    for (rebin = 0 ; rebin < rt_detector_rebin_rows - 1; rebin++) {

      /* reuse previous values */

      cycle_next(rebin_coords);
      cycle_next(fwd_rebin_row);

      rebin_coords[next] = (-rt_pi / 2.0f) - rt_half_fan_angle + 
          rt_detector_rebin_rows_height * float(rebin + 1);

      fwd_rebin_row[next] = rebin_scale * (rebin_coords[next] +
          (rebin_coords[next] /	tanf(rebin_coords[next]))	*
          (col_coord / rt_scan_diameter));

      if (row_coord >= fwd_rebin_row[cur] &&	
          row_coord <= fwd_rebin_row[next]) {
            rebin_row = rebin;

            fracs[0] = (row_coord - fwd_rebin_row[cur]) / 
                (fwd_rebin_row[next] - fwd_rebin_row[cur]);

            fracs[1] -= fracs[0];            
            break;
      }
    }
    offset = rebin_offset(proj_local, rebin_row, rebin_col);
    result = fracs[1] * gpu_input[offset];

    offset = rebin_offset(proj_local, rebin_row+1, rebin_col);
    result += fracs[0] * gpu_input[offset];

    offset = pixel_offset(proj_local, row, col);

    gpu_output[offset] = result;

  } else {

    /* column coordinate in negative range */
    /* Find the rebin row index that fits limits (one as default) */

    rebin_row = 1;
    rebin_col = col;
    fracs[0] = 0.0f; fracs[1] = 1.0f;

    /* unrolled warm up round for loop - prepares cur values for first loop */

    rebin = rt_detector_rebin_rows;
    rebin_coords[prev] = (-rt_pi / 2.0f) - rt_half_fan_angle + 
        rt_detector_rebin_rows_height * float(rebin - 1);

    fwd_rebin_row[prev] = rebin_scale * (rebin_coords[prev] +	
        (rebin_coords[prev] / tanf(rebin_coords[prev])) *
        (col_coord / rt_scan_diameter));

    for (rebin = rt_detector_rebin_rows - 1; rebin > 0; rebin--) {

      /* reuse previous values */

      cycle_prev(rebin_coords);
      cycle_prev(fwd_rebin_row);

      rebin_coords[prev] = (-rt_pi / 2.0f) - rt_half_fan_angle + 
          rt_detector_rebin_rows_height * float(rebin - 1);

      fwd_rebin_row[prev] = rebin_scale * (rebin_coords[prev] +		
          (rebin_coords[prev] / tanf(rebin_coords[prev])) *
          (col_coord / rt_scan_diameter));

      if (row_coord >= fwd_rebin_row[prev] &&	
            row_coord <= fwd_rebin_row[cur]) {
            rebin_row = rebin;
            rebin_col = col;

	          fracs[0] = (row_coord -	fwd_rebin_row[prev]) /
                (fwd_rebin_row[cur] - fwd_rebin_row[prev]);

            fracs[1] -= fracs[0];
            break;
      }
    }
    offset = rebin_offset(proj_local, rebin_row-1, rebin_col);
    result = fracs[1] * gpu_input[offset];

    offset = rebin_offset(proj_local, rebin_row, rebin_col);
    result += fracs[0] * gpu_input[offset];

    offset = pixel_offset(proj_local, row, col);

    gpu_output[offset] = result;
  }
}


/* 
   Back projection of individual (x, y, z) voxels with projections in range
   from first_proj to last_proj. Each thread handles one voxel and blocks are
   launched to cover entire chunk in FoV.
*/

KERNEL void flat_backproject_chunk(const int chunk_index,
				   const int first_proj,
				   const int last_proj,
				   const int first_z,
				   const int last_z,
				   GLOBALMEM float *gpu_input, 
				   GLOBALMEM float *proj_row_mins,
				   GLOBALMEM float *proj_row_maxs,
				   GLOBALMEM float *gpu_output) {

  /* please note that we should match adjacent threads to adjacent z voxels */

  int x_local = thread_backproject_x();
  int y_local = thread_backproject_y();
  int z_local = thread_backproject_z();
  
  int proj_index;

  /* global indices */

  int x, y, z;
  float x_coord, y_coord, z_coord;

  /* triple projection helpers */

  float scale_help[3], z_coord_min[3], z_coord_max[3], proj_col_frac[3];
  float proj_row_coord[3], proj_row_coord_min[3], proj_row_coord_max[3];
  int proj[3], proj_col_int[3], z_first[3], z_last[3];

  /* single projection helpers */

  float source_angle = -1.0f, proj_col_real = -1.0f, proj_col_coord = -1.0f;
  float proj_row_real = -1.0f, proj_row_frac = -1.0f;
  int proj_row_int = -1, offset = -1;
  float weight = 0.0f, contrib = 0.0f, weighted_contrib = 0.0f, result = 0.0f;

  /* helpers */
  /* global indices */

  x = x_local;
  y = y_local;
  z = z_local + first_z;
  x_coord = rt_x_min + float(x) * rt_delta_x;
  y_coord = rt_y_min + float(y) * rt_delta_y;
  z_coord = rt_z_min + float(z) * rt_delta_z;

  if (z > last_z) {
    //printf("kernel thread with z out of range: %d\n", z);
    return;
  }

  /* Only reconstruct center cylinder */

  if (x_coord * x_coord + y_coord * y_coord > rt_fov_radius*rt_fov_radius)
    return;

  /*
    Iterate through all projections in chunk and add up interpolated pixel
    values for those that contribute to this particular voxel

    We need calculations for previous and next projection for boundary weight
    so we add two extra rounds to gather information.

    Unrolling the boundary loops does not seem to improve performance.
  */

  for (proj_index=first_proj-1; proj_index <= last_proj+1; proj_index++) {

    /* cycle values to prepare for new next proj */

    cycle_next(proj);
    cycle_next(scale_help);
    cycle_next(proj_col_int);
    cycle_next(proj_col_frac);
    cycle_next(proj_row_coord_min);
    cycle_next(proj_row_coord_max);
    cycle_next(proj_row_coord);
    cycle_next(z_coord_min);
    cycle_next(z_coord_max);
    cycle_next(z_first);
    cycle_next(z_last);

    /* calculate helpers for next projection */

    proj[next] = proj_index - first_proj;
    source_angle = rt_s_min +  rt_delta_s * (float(proj_index) + 0.5f);

    /* scale helper and column coordinate from projection formula */

    scale_help[next] = rt_scan_radius - x_coord * cosf(source_angle) - 
        y_coord * sinf(source_angle);

    proj_col_coord = (rt_scan_diameter / scale_help[next]) * \
      (-x_coord * sinf(source_angle) + y_coord * cosf(source_angle));

    /* translate absolute column coordinate to pixel index */
    /* sign is inverted for shift and thus also for offset */

    proj_col_real = proj_col_coord / rt_detector_pixel_span + 0.5f * \
      float(rt_detector_columns) - rt_detector_column_offset;

    /* integer and fractional part for interpolation */

    proj_col_int[next] = (int)proj_col_real;

    /* Make sure col is in valid range for row boundary interpolation */

    proj_col_int[next] = min(max(proj_col_int[next], 0), 
        rt_detector_columns - 1);

    proj_col_frac[next] = proj_col_real - float(proj_col_int[next]);

    /* interpolate closest precalculated Tam-Danielsson window row borders */

    proj_row_coord_min[next] = (1.0f - proj_col_frac[next]) * 
      proj_row_mins[proj_col_int[next]] + proj_col_frac[next] * 
      proj_row_mins[proj_col_int[next] + 1];

    proj_row_coord_max[next] = (1.0f - proj_col_frac[next]) * 
      proj_row_maxs[proj_col_int[next]] + proj_col_frac[next] * 
      proj_row_maxs[proj_col_int[next] + 1];

    /* Find the row coordinate for this z using projection formula */ 

    proj_row_coord[next] = (rt_scan_diameter / scale_help[next]) * 
      (z_coord - rt_progress_per_radian * source_angle);

    /* Find the z coordinate boundaries from the projection formula */

    z_coord_min[next] = source_angle * rt_progress_per_radian + 
      proj_row_coord_min[next] * scale_help[next] / rt_scan_diameter;

    z_coord_max[next] = source_angle * rt_progress_per_radian + 
      proj_row_coord_max[next] * scale_help[next] / rt_scan_diameter;

    /* translate to z border indices */

    z_first[next] = ceil((z_coord_min[next] - rt_z_min) / rt_delta_z);
    z_last[next] = floor((z_coord_max[next] - rt_z_min) / rt_delta_z);

    /* Stop here if first two warm up rounds or if no contribution */

    if (proj_index <= first_proj || z < z_first[cur] || z > z_last[cur])
      continue;

    if (z == z_first[cur] && proj_row_coord[next] < proj_row_coord_min[next]) {
        weight = 0.5f + (z_coord - z_coord_min[cur]) / 
            (z_coord_min[next] - z_coord_min[cur]);

    } else if (z == z_last[cur] &&
          proj_row_coord[prev] > proj_row_coord_max[prev]) {
              weight = 0.5f + (z_coord_max[cur] - z_coord) / 
                  (z_coord_max[cur] - z_coord_max[prev]);

    } else {
        weight = 1.0f;
    }

    /* Make sure col+1 is in valid range */

    proj_col_int[cur] = min(max(proj_col_int[cur], 0), 
        rt_detector_columns - 2);

    /* sign is inverted for shift and thus also for offset */

    proj_row_real = proj_row_coord[cur] / rt_detector_pixel_height + 
        0.5f * float(rt_detector_rows - 1) - rt_detector_row_offset;

    proj_row_int = (int)proj_row_real;

    /* Make sure row+1 is in valid range */

    proj_row_int = min(max(proj_row_int, 0), rt_detector_rows - 2);
    proj_row_frac = proj_row_real - float(proj_row_int);

    /* Manually interpolate four nearest pixels in projection */

    contrib = 0.0f;
    offset = pixel_offset(proj[cur], proj_row_int, proj_col_int[cur]);    
    contrib += (1.0f - proj_row_frac) * (1.0f - proj_col_frac[cur]) * 
        gpu_input[offset];

    offset = pixel_offset(proj[cur], proj_row_int+1, proj_col_int[cur]);
    contrib += proj_row_frac * (1.0f - proj_col_frac[cur]) * 
        gpu_input[offset];

    offset = pixel_offset(proj[cur], proj_row_int, proj_col_int[cur]+1);
    contrib += (1.0f - proj_row_frac) *  proj_col_frac[cur] * 
        gpu_input[offset];

    offset = pixel_offset(proj[cur], proj_row_int+1, proj_col_int[cur]+1);
    contrib += proj_row_frac * proj_col_frac[cur] * gpu_input[offset];

    weighted_contrib = (weight / scale_help[cur]) * contrib;

    result += weighted_contrib;
  }
  offset = voxel_offset(x, y, z_local);
  gpu_output[offset] += (result / float(rt_projs_per_turn));

  return;
}


/*
  Differentiate individual pixels in chunk of projections. Each thread handles
  one pixel and blocks are launched to cover entire chunk in projections.
 */

KERNEL void curved_diff_chunk(const int first,
			      const int last,
			      GLOBALMEM float *gpu_input,
			      GLOBALMEM float *gpu_output) {

  /* please note that order in block is reversed to col, row, proj */

  int proj_local = thread_filtering_proj_local();
  int row = thread_filtering_row();
  int col = thread_filtering_col();
  int proj = proj_local + first;
  int offsets[9];

  float d_proj = -1.0f, d_col = -1.0f;
  float dia = -1.0f, dia_sqr = -1.0f, result = 0.0f;
  float row_coords, row_sqr;

  /* skip last index in each dimension to avoid index+1 out of bounds */

  if (proj >= last || 
      row >= rt_detector_rows - 1 || 
      col >= rt_detector_columns - 1)
        return;

  dia = rt_scan_diameter;
  dia_sqr = dia * dia;

  /* 
     Detector pixel center coordinates using coordinate system centered in
     the middle of the detector. 
     Please note that the row coordinate formula in (82) of the Noo paper        
     is wrong. It should be Nrows instead of Ncols and fig 4 indicates that      
     the center row coordinates are used which means it should include the       
     '-1' too to get the half pixel offset like for the curved detector.         
  */

  row_coords = rt_detector_pixel_height * 
      (float(row) + rt_detector_row_offset - 
      0.5f * float(rt_detector_rows - 1));

  row_sqr = row_coords * row_coords;

  /* Differentiate using the chain rule and manual interpolation */

  /* extract neighbor pixel values using the naming scheme described above */

  offsets[cur_sw] = pixel_offset(proj_local, row, col);
  offsets[cur_se] = pixel_offset(proj_local, row, col+1);
  offsets[next_sw] = pixel_offset(proj_local+1, row, col);
  offsets[next_se] = pixel_offset(proj_local+1, row, col+1);

  d_proj = (gpu_input[offsets[next_sw]] - gpu_input[offsets[cur_sw]] +	
      gpu_input[offsets[next_se]] - gpu_input[offsets[cur_se]]) / 
      (2.0f * rt_delta_s);

  d_col = (gpu_input[offsets[cur_se]] - gpu_input[offsets[cur_sw]] +	
      gpu_input[offsets[next_se]] - gpu_input[offsets[next_sw]]) / 
      (2.0f * rt_detector_pixel_span);

  result = d_proj + d_col;
  result *= dia / sqrt(dia_sqr + row_sqr);

  /* out of bounds check */
  /*
  if (offsets[cur_sw] >= (last-first)*rt_detector_rows*rt_detector_columns) {
    printf("ERROR: diff_chunk out of bounds! %d\n", offsets[cur_sw]);
    return;
  }
  */

  gpu_output[offsets[cur_sw]] = result;
}


/* 
   Forward rebin individual pixels in chunk of projections. Each thread handles
   one pixel and blocks are launched to cover entire chunk in rebinned
   projections.
 */

KERNEL void curved_fwd_rebin_chunk(const int first,
				   const int last,
				   GLOBALMEM float *gpu_input,
				   GLOBALMEM float *gpu_output) {

  /* please note that order in block is reversed to col, rebin_row, proj */

  int proj_local = thread_filtering_proj_local();
  int rebin_row = thread_filtering_rebin_row();
  int col = thread_filtering_col();
  int proj = proj_local + first;

  int row = -1, offset = -1;
  float fwd_rebin_row = -1.0f, row_scaled = -1.0f, row_frac = -1.0f;
  float rebin_scale = -1.0f, rebin_coord = -1.0f, col_coord = -1.0f;
  float result = 0.0f;

  /* skip last index in projs and cols */

  if (proj >= last || 
      rebin_row > rt_detector_rebin_rows || 
      col >= rt_detector_columns)
        return;

  rebin_scale = 2.0f * rt_progress_per_radian;

  rebin_coord = (-rt_pi / 2.0f) - rt_half_fan_angle + 
      rt_detector_rebin_rows_height * float(rebin_row);

  col_coord = rt_detector_pixel_span * 
      (col + rt_detector_column_offset - 
      0.5f * float(rt_detector_columns - 1));

  /* TODO: we could maybe gain from using sincosf here */

  fwd_rebin_row = rebin_scale * (rebin_coord * cosf(col_coord) + 
				 (rebin_coord /	tanf(rebin_coord)) * 
				 sinf(col_coord));

  /* sign is inverted for shift and thus also for offset */

  row_scaled = fwd_rebin_row / rt_detector_pixel_height + 
      0.5f * float(rt_detector_rows) - rt_detector_row_offset;

  /* make sure row and row+1 are in valid row range */

  row_scaled = min(max(0.0f, row_scaled), float(rt_detector_rows) - 2.0f);
  row = (int)row_scaled;
  row_frac = row_scaled - float(row);

  /* Interpolate the two row neighbors */

  offset = pixel_offset(proj_local, row, col);
  result = (1.0f - row_frac) * gpu_input[offset];

  offset = pixel_offset(proj_local, row+1, col);
  result += row_frac * gpu_input[offset];

  offset = rebin_offset(proj_local, rebin_row, col);

  /* out of bounds check */
  /*
  if (offset >= (last-first)*rt_detector_rebin_rows*rt_detector_columns) {
    printf("ERROR: fwd_rebin_chunk out of bounds! %d\n", offset);
    return;
  }
  */

  gpu_output[offset] = result;
}


/* 
   Convolution of individual pixels in chunk of rebinned projections. Each
   thread handles one pixel and blocks are launched to cover entire chunk in 
   rebinned projections.
 */

KERNEL void curved_convolve_chunk(const int first,
				  const int last,
				  GLOBALMEM float *gpu_input,
				  GLOBALMEM float *hilbert_ideal,
				  GLOBALMEM float *gpu_output) {

  /* please note that order in block is reversed to col, rebin_row, proj */

  int proj_local = thread_filtering_proj_local();
  int rebin_row = thread_filtering_rebin_row();
  int col = thread_filtering_col();

  int offset = -1, conv_col = -1, i = -1;
  float result = 0.0f;

  SHAREDMEM float shared_hilbert_ideal[rt_kernel_width];

  /* Prefetch hilbert ideal to SHAREDMEM */

  offset = local_offset(GET_LOCAL_ID_X, GET_LOCAL_ID_Y, GET_LOCAL_ID_Z);
  while (offset < rt_kernel_width) {     
    shared_hilbert_ideal[offset] = hilbert_ideal[offset];
    offset += local_elems;
  }
  
  sync_shared_mem();

  /*
    TODO: use rectangular hilbert window as suggested in Noo paper?
  */
  /* 
     convolve(f, g)[n] is defined to be the sum of f[m]*g[n-m] for all m
     in range (-inf, inf).
     We only need the central slice of the resulting N+M-1 discrete 
     elements.
  */

  for (i = -rt_kernel_radius; i <= rt_kernel_radius; i++) {

    /* We reverse sum order to step through rows in a cache efficient way */

    conv_col = col + i;

    /* make sure conv_col index stays in rebin_row */

    conv_col = min(max(conv_col, 0), rt_detector_columns - 1);

    offset = rebin_offset(proj_local, rebin_row, conv_col);

    /* reversed hilbert access order to match reversed col order above */

    result += shared_hilbert_ideal[rt_kernel_radius - i] * gpu_input[offset];
  }

  offset = rebin_offset(proj_local, rebin_row, col);
  gpu_output[offset] = result;
}


/* 
   Reverse rebin individual pixels in chunk of projections. Each thread handles
   one pixel and blocks are launched to cover entire chunk in projections.
*/

KERNEL void curved_rev_rebin_chunk(const int first,
				   const int last,
				   GLOBALMEM float *gpu_input,
				   GLOBALMEM float *gpu_output) {

  /* please note that order in block is reversed to col, row, proj */

  int proj_local = thread_filtering_proj_local();
  int row = thread_filtering_row();
  int col = thread_filtering_col();

  int rebin_row = -1, rebin_col = -1, rebin = -1, offset = -1;

  /* We allocate three rebin elements to use easy-read prev, cur, next */

  float fracs[2], fwd_rebin_row[3], rebin_coords[3];
  float row_coord = -1.0f, col_coord = -1.0f, rebin_scale = -1.0f;

  float result = 0.0f;

  row_coord = rt_detector_pixel_height * 
      (float(row) + rt_detector_row_offset - 
      0.5f * float(rt_detector_rows - 1));

  col_coord = rt_detector_pixel_span * 
      (float(col) + rt_detector_column_offset - 
      0.5f * float(rt_detector_columns - 1));

  rebin_scale = 2.0f * rt_progress_per_radian;

  /* col offset ruins positive/negative half split */

  if (col_coord >= 0.0f) {

    /* column coordinate in positive range */

    rebin_row = 0;
    rebin_col = col;
    fracs[0] = 0.0f; fracs[1] = 1.0f;

    /* unrolled warm up round for loop - prepares cur values for first loop */

    rebin = -1;
    rebin_coords[next] = (-rt_pi / 2.0f) - rt_half_fan_angle + 
        rt_detector_rebin_rows_height * float(rebin + 1);

    /* TODO: we could maybe gain from using sincosf here */

    fwd_rebin_row[next] = rebin_scale * (rebin_coords[next] * cosf(col_coord) +
        (rebin_coords[next] /	tanf(rebin_coords[next])) * sinf(col_coord));

    for (rebin = 0 ; rebin < rt_detector_rebin_rows - 1; rebin++) {

      /* reuse previous values */

      cycle_next(rebin_coords);
      cycle_next(fwd_rebin_row);

      rebin_coords[next] = (-rt_pi / 2.0f) - rt_half_fan_angle + 
          rt_detector_rebin_rows_height * float(rebin + 1);

      /* TODO: we could maybe gain from using sincosf here */

      fwd_rebin_row[next] = rebin_scale * (rebin_coords[next] *	
          cosf(col_coord) + (rebin_coords[next] / tanf(rebin_coords[next])) *
					sinf(col_coord));

      if (row_coord >= fwd_rebin_row[cur] &&	\
          row_coord <= fwd_rebin_row[next]) {
              rebin_row = rebin;
              fracs[0] = (row_coord - fwd_rebin_row[cur]) / 
                  (fwd_rebin_row[next] - fwd_rebin_row[cur]);

              fracs[1] -= fracs[0];
              break;
      }
    }
    offset = rebin_offset(proj_local, rebin_row, rebin_col);
    result = fracs[1] * gpu_input[offset];

    offset = rebin_offset(proj_local, rebin_row+1, rebin_col);
    result += fracs[0] * gpu_input[offset];

    offset = pixel_offset(proj_local, row, col);

    /* Curved detector requires cosinus weighting of result */

    result *= cosf(col_coord);
    gpu_output[offset] = result;

  } else {

    /* column coordinate in negative range */
    /* Find the rebin row index that fits limits (one as default) */

    rebin_row = 1;
    rebin_col = col;
    fracs[0] = 0.0f; fracs[1] = 1.0f;

    /* unrolled warm up round for loop - prepares cur values for first loop */

    rebin = rt_detector_rebin_rows;
    rebin_coords[prev] = (-rt_pi / 2.0f) - rt_half_fan_angle + 
        rt_detector_rebin_rows_height * float(rebin - 1);

    /* TODO: we could maybe gain from using sincosf here */

    fwd_rebin_row[prev] = rebin_scale * (rebin_coords[prev] * cosf(col_coord)
        + (rebin_coords[prev] / tanf(rebin_coords[prev])) * sinf(col_coord));

    for (rebin = rt_detector_rebin_rows - 1; rebin > 0; rebin--) {

      /* reuse previous values */

      cycle_prev(rebin_coords);
      cycle_prev(fwd_rebin_row);

      rebin_coords[prev] = (-rt_pi / 2.0f) - rt_half_fan_angle + 
          rt_detector_rebin_rows_height * float(rebin - 1);

      /* TODO: we could maybe gain from using sincosf here */

      fwd_rebin_row[prev] = rebin_scale * (rebin_coords[prev] * 
          cosf(col_coord) + (rebin_coords[prev] / tanf(rebin_coords[prev])) *	
          sinf(col_coord));

      if (row_coord >= fwd_rebin_row[prev] &&	\
          row_coord <= fwd_rebin_row[cur]) {
              rebin_row = rebin;
              rebin_col = col;

              fracs[0] = (row_coord -	fwd_rebin_row[prev]) / 
                  (fwd_rebin_row[cur] - fwd_rebin_row[prev]);

              fracs[1] -= fracs[0];
              break;
      }
    }

    offset = rebin_offset(proj_local, rebin_row-1, rebin_col);
    result = fracs[1] * gpu_input[offset];

    offset = rebin_offset(proj_local, rebin_row, rebin_col);
    result += fracs[0] * gpu_input[offset];

    offset = pixel_offset(proj_local, row, col);

    /* Curved detector requires cosinus weighting of result */

    result *= cosf(col_coord);
    gpu_output[offset] = result;
  }
}


/* 
   Back projection of individual (x, y, z) voxels with projections in range
   from first to last. Each thread handles one voxel and blocks are launched
   to cover entire chunk in FoV.
*/

KERNEL void curved_backproject_chunk(const int chunk_index,
				     const int first_proj, 
				     const int last_proj,
				     const int first_z,
				     const int last_z, 
				     GLOBALMEM float *gpu_input, 
				     GLOBALMEM float *proj_row_mins,
				     GLOBALMEM float *proj_row_maxs,
				     GLOBALMEM float *gpu_output) {

  /* please note that we should match adjacent threads to adjacent z voxels */

  int x_local = thread_backproject_x();
  int y_local = thread_backproject_y();
  int z_local = thread_backproject_z();

  int proj_index;

  /* global indices */

  int x, y, z;
  float x_coord, y_coord, z_coord;

  /* triple projection helpers */

  /*
     NOTE: We don't actually use all the array values but it is *not* faster to
     have only the ones needed as ordinary variables, or using cyclic index
     instead of this swapping values and keeping prev/cur/next index static.
     For some reason it runs quite a lot slower in OpenCL than in CUDA on NVidia
     (~50%) but it's scales on AMD.
  */

  float scale_help[3], z_coord_min[3], z_coord_max[3], proj_col_frac[3];
  float proj_row_coord[3], proj_row_coord_min[3], proj_row_coord_max[3];
  int proj[3], proj_col_int[3], z_first[3], z_last[3];

  /* single projection helpers */

  float source_angle = -1.0f, proj_col_real = -1.0f, proj_col_coord = -1.0f;
  float proj_row_real = -1.0f, proj_row_frac = -1.0f, cos_proj_col_coord;
  int proj_row_int = -1, offset = -1;
  float weight = 0.0f, contrib = 0.0f, weighted_contrib = 0.0f, result = 0.0f;

  /* helpers */
  /* global indices */
  x = x_local;
  y = y_local;
  z = z_local + first_z;
  x_coord = rt_x_min + float(x) * rt_delta_x;
  y_coord = rt_y_min + float(y) * rt_delta_y;
  z_coord = rt_z_min + float(z) * rt_delta_z;

  if (z > last_z) {
    //printf("kernel thread with z out of range: %d\n", z);
    return;
  }

  /* Only reconstruct center cylinder */

  if (x_coord*x_coord + y_coord*y_coord > rt_fov_radius*rt_fov_radius)
    return;

  /*
    Iterate through all projections in chunk and add up interpolated pixel
    values for those that contribute to this particular voxel

    We need calculations for previous and next projection for boundary weight
    so we add two extra rounds to gather information.

    Unrolling the boundary loops does not seem to improve performance.
  */

  for (proj_index=first_proj-1; proj_index <= last_proj+1; proj_index++) {

    /* cycle values to prepare for new next proj */

    cycle_next(proj);
    cycle_next(scale_help);
    cycle_next(proj_col_int);
    cycle_next(proj_col_frac);
    cycle_next(proj_row_coord_min);
    cycle_next(proj_row_coord_max);
    cycle_next(proj_row_coord);
    cycle_next(z_coord_min);
    cycle_next(z_coord_max);
    cycle_next(z_first);
    cycle_next(z_last);

    /* TODO: don't calculate unnecessary next vars in last iteration */

    /* calculate helpers for next projection */

    proj[next] = proj_index - first_proj;
    source_angle = rt_s_min +  rt_delta_s * (float(proj_index) + 0.5f);

    /* scale helper and column coordinate from projection formula */

    /* TODO: we could maybe gain from using sincosf here */

    scale_help[next] = rt_scan_radius - x_coord * cosf(source_angle) -
        y_coord * sinf(source_angle);

    /* TODO: we could maybe gain from using sincosf here */

    proj_col_coord = atanf((1.0f / scale_help[next]) * 	
        (-x_coord * sinf(source_angle) + y_coord *
        cosf(source_angle)));

    cos_proj_col_coord = cosf(proj_col_coord);

    /* translate absolute column coordinate to pixel index */
    /* sign is inverted for shift and thus also for offset */

    proj_col_real = proj_col_coord / rt_detector_pixel_span + 
        0.5f * float(rt_detector_columns) - rt_detector_column_offset;

    /* integer and fractional part for interpolation */

    proj_col_int[next] = (int)proj_col_real;

    /* Make sure col is in valid range for row boundary interpolation */

    proj_col_int[next] = min(max(proj_col_int[next], 0), 
        rt_detector_columns - 1);

    proj_col_frac[next] = proj_col_real - float(proj_col_int[next]);

    /* interpolate closest precalculated Tam-Danielsson window row borders */

    proj_row_coord_min[next] = (1.0f - proj_col_frac[next]) * 
      proj_row_mins[proj_col_int[next]] + proj_col_frac[next] * 
      proj_row_mins[proj_col_int[next] + 1];

    proj_row_coord_max[next] = (1.0f - proj_col_frac[next]) * 
      proj_row_maxs[proj_col_int[next]] + proj_col_frac[next] * 
      proj_row_maxs[proj_col_int[next] + 1];

    /* Find the row coordinate for this z using projection formula */ 

    proj_row_coord[next] = (rt_scan_diameter * cos_proj_col_coord / 
        scale_help[next]) *
        (z_coord - rt_progress_per_radian * source_angle);

    /* Find the z coordinate boundaries from the projection formula */

    z_coord_min[next] = source_angle * rt_progress_per_radian +
        proj_row_coord_min[next] * scale_help[next] / 
        (rt_scan_diameter * cos_proj_col_coord);

    z_coord_max[next] = source_angle * rt_progress_per_radian + 
      proj_row_coord_max[next] * scale_help[next] / (rt_scan_diameter * 
						     cos_proj_col_coord);

    /* translate to z border indices */

    z_first[next] = ceil((z_coord_min[next] - rt_z_min) / rt_delta_z);
    z_last[next] = floor((z_coord_max[next] - rt_z_min) / rt_delta_z);

    /* Stop here if first two warm up rounds or if no contribution */

    if (proj_index <= first_proj || z < z_first[cur] || z > z_last[cur])
      continue;

    if (z == z_first[cur] && proj_row_coord[next] < proj_row_coord_min[next]) {
        weight = 0.5f + (z_coord - z_coord_min[cur]) / 
        (z_coord_min[next] - z_coord_min[cur]);

    } else if (z == z_last[cur] &&
          proj_row_coord[prev] > proj_row_coord_max[prev]) {
              weight = 0.5f + (z_coord_max[cur] - z_coord) / 
                  (z_coord_max[cur] - z_coord_max[prev]);

    } else {
        weight = 1.0f;
    }

    /* Make sure col+1 is in valid range */

    proj_col_int[cur] = min(max(proj_col_int[cur], 0), 
        rt_detector_columns - 2);

    /* sign is inverted for shift and thus also for offset */

    proj_row_real = proj_row_coord[cur] / rt_detector_pixel_height + 
        0.5f * float(rt_detector_rows - 1) - rt_detector_row_offset;

    proj_row_int = (int)proj_row_real;

    /* Make sure row+1 is in valid range */

    proj_row_int = min(max(proj_row_int, 0), rt_detector_rows - 2);
    proj_row_frac = proj_row_real - float(proj_row_int);

    /* Manually interpolate four nearest pixels in projection */
    contrib = 0.0;
    offset = pixel_offset(proj[cur], proj_row_int, proj_col_int[cur]);
    contrib += (1.0f - proj_row_frac) * (1.0f - proj_col_frac[cur]) * 
        gpu_input[offset];

    offset = pixel_offset(proj[cur], proj_row_int+1, proj_col_int[cur]);
    contrib += proj_row_frac * (1.0f - proj_col_frac[cur]) * 
        gpu_input[offset];

    offset = pixel_offset(proj[cur], proj_row_int, proj_col_int[cur]+1);
    contrib += (1.0f - proj_row_frac) *  proj_col_frac[cur] * \
        gpu_input[offset];

    offset = pixel_offset(proj[cur], proj_row_int+1, proj_col_int[cur]+1);
    contrib += proj_row_frac * proj_col_frac[cur] * gpu_input[offset];

    weighted_contrib = (weight / scale_help[cur]) * contrib;
    result += weighted_contrib;
  }
  
  offset = voxel_offset(x, y, z_local);
  gpu_output[offset] += (result / float(rt_projs_per_turn));

  return;
}


/*
  Kernel to validate device array
*/
KERNEL void checksum_array(GLOBALMEM float *result, 
			   GLOBALMEM float *arr,
			   const int first,
			   const int last)
{
  int tid = THREAD_ID_X + THREAD_ID_Y * + BLOCK_DIM_X; 
  int i = -1;
  const float scale = 0.0000001f;

  if (tid == 0) {  
    result[0] = 0.0f;
    for (i = first; i < last; i++) {
      // include index to catch cases with same values but in different layout
      //result[0] += arr[i];

      result[0] += (1.0+scale*i)*(arr[i]+scale);
    }
  }
}
