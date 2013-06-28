/*
#
# --- BEGIN_HEADER ---
#
# kernels - FDK cone beam reconstruction CUDA Kernels
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

/*
Pure kernels for circular cone beam CT in CUDA using the FDK algorithm

Requires external init of geometry configuration like scan_radius and so on.
This may come from init.cu or be hard coded from runtime generated code.
*/


/* Constants */

#define TRANSFORM_MATRIX_SIZE 12
#define Z_SLICE_SIZE (rt_y_voxels*rt_x_voxels)
#define SOURCE_DETECTOR_DISTANCE (rt_source_distance+rt_detector_distance)


/* Macros fixing CUDA quirks */

#define pow2f(x) powf(abs(x), 2)


/* Flat indexing macros */

#define COMPLEX_PROJ_REAL_IDX(y,x) (((y)*rt_proj_filter_width*2)+(x*2))
#define COMPLEX_PROJ_IDX(y,x) (((y)*rt_proj_filter_width*2)+(x))
#define PROJ_IDX(y,x) ((y)*rt_detector_columns+(x))
#define VOLUME_SLICE_IDX(y,x) (((y)*rt_x_voxels)+(x))


/* Weight projection data */

__global__ void weight_proj(float *proj,
			    unsigned int *proj_row_offset,
			    float *weight) {

   unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;   
   unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
      
   proj[PROJ_IDX(y,x)] *= weight[PROJ_IDX(y+*proj_row_offset,x)];
}


/* 
 * Convert float projection matrix to complex matrix
 * with the number of columns matching the projection filter width.
 * The complex projection is used when filteren projection data
 * in the frequency domain.
 * We represent a complex value as two contigours float values,
 * one for the real part and one for the imaginary part.
 */

__global__ void proj_to_complex(float *complex_proj,
				float *float_proj) {

   unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;   
   unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
      
   complex_proj[COMPLEX_PROJ_REAL_IDX(y,x)] = float_proj[PROJ_IDX(y,x)];
}


/* 
 * The projection is in the complex frequence domain at this point,
 * this means it has an real and image part each represented as a float.
 * The length of the complex frequency projection is therefore
 * two times the projection filter width
 */

__global__ void filter_proj(float *complex_proj,
			    float *filter) {

   unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;   
   unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
       
   // x is the index for the complex frequency projection
   // We divede by two in order to get the filter index

   int filter_idx = x>>1;

   complex_proj[COMPLEX_PROJ_IDX(y,x)] *= filter[filter_idx];
}


/* 
 * Convert complex projection to float projection 
 * (see proj_to_complex for more details)
 */

__global__ void complex_to_proj(float *float_proj,
				float *complex_proj) {

   unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;   
   unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
      
   float_proj[PROJ_IDX(y,x)] = complex_proj[COMPLEX_PROJ_REAL_IDX(y,x)];
}


/* Generate volume weight based on projection angle and voxel coordinates */

__global__ void generate_volume_weight(float *volume_weight_matrix,
				       float *combined_matrix,
				       float *proj_angle_rad) {
   
   unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
   unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

   float sincos[2];
   float combined[2];
   
   combined[0] = combined_matrix[VOLUME_SLICE_IDX(y,x)];
   combined[1] = combined_matrix[VOLUME_SLICE_IDX(y,x) + Z_SLICE_SIZE];

   sincosf(*proj_angle_rad, &sincos[0], &sincos[1]);
     
#ifdef rt_detector_shape_flat
   volume_weight_matrix[VOLUME_SLICE_IDX(y,x)] = 
     (pow2f(rt_source_distance) /
	 pow2f(rt_source_distance
	       - sincos[1] * combined[0]
	       - sincos[0] * combined[1]))
     * rt_volume_weight_factor;
#else
   volume_weight_matrix[VOLUME_SLICE_IDX(y,x)] = 
     (pow2f(rt_source_distance) /
	 (pow2f(rt_source_distance 
		- (sincos[1] * combined[0]
		   +  sincos[0] * combined[1]))
	   + pow2f(-sincos[0] * combined[0]
		   + sincos[1] * combined[1])))
     * rt_volume_weight_factor;
#endif
}


/*
 * This function backprojects one voxel from the projection data,
 */

__forceinline__ __device__ void backproject_slice(float *recon_voxel,
                                                  float *transform_matrix, 
                                                  float *combined,
                                                  float *proj_data,
						  unsigned int *proj_row_offset) {

   // First calculate the dot product between transform and combined

   float dot_row0;
   float dot_row1;
   float dot_row2;
      
   dot_row0 = (transform_matrix[0] * combined[0])
            + (transform_matrix[1] * combined[1])
            + (transform_matrix[2] * combined[2])
	    + (transform_matrix[3]);
   
   dot_row1 = (transform_matrix[4] * combined[0])
            + (transform_matrix[5] * combined[1])
            + (transform_matrix[6] * combined[2])
            + (transform_matrix[7]);

   dot_row2 = (transform_matrix[8] * combined[0])
            + (transform_matrix[9] * combined[1])
            + (transform_matrix[10] * combined[2])
            + (transform_matrix[11]);
   
   // Find the projection pixel index that maps to the backprojected voxel

   unsigned char mask;
   int map_col, map_row;

   float flat_map_col = dot_row0/dot_row2;
   float flat_map_row = dot_row1/dot_row2;

#ifdef rt_detector_shape_flat
   map_col = int(rint(flat_map_col));
   map_row = int(rint(flat_map_row));
#else
   float alpha = atanf(flat_map_col/float(SOURCE_DETECTOR_DISTANCE));
         
   map_col = int(rintf(alpha 
		       * float(SOURCE_DETECTOR_DISTANCE) 
		       * float(1.0f/rt_detector_pixel_width) 
		       + float(rt_detector_column_shift)));
           
   map_row = int(rintf(flat_map_row 
		       * cosf(alpha) 
		       * float(1.0f/rt_detector_pixel_height) 
		       + float(rt_detector_row_shift)));
#endif

   
   // Mask out voxel if the ray passing through it doensn't hit the detector

   mask = (map_col>=0) & (map_col<rt_detector_columns) 
     & (map_row>=0) & (map_row<rt_detector_rows);

   
   // Offset map_row according to projection row offset for the processed chunk

   map_row -= *proj_row_offset;
   
   *recon_voxel = mask ? proj_data[PROJ_IDX(map_row, map_col)] : 0.0f;
}



/* 
 * This function loops over the z slices in recon_chunk and 
 * backprojects the projection data for each slice 
 */

__global__ void backproject(float *recon_chunk,
			    float *proj_data,
			    unsigned int *proj_row_offset,
			    float *proj_angle_rad,
			    unsigned int *chunk_index,
			    float *z_voxel_coordinates,
			    float *transform_matrix,
			    float *combined_matrix,
			    float *volume_weight_matrix) {

   unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
   unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
   
   // Cache values z-voxel coordinates re-used in each for loop iteration

   __shared__ float shared_z_voxel_coordinates[rt_z_voxels];

   int btidx = threadIdx.y * blockDim.x + threadIdx.x;
   int shared_index = btidx;
   
   shared_index = btidx;
   while (shared_index < rt_z_voxels) {
      shared_z_voxel_coordinates[shared_index] = 
	z_voxel_coordinates[shared_index];
      
      shared_index += blockDim.x*blockDim.y;
   }

   __syncthreads();

   
   // Initialize register variables

   float recon_voxel;
   
#ifndef volume_weight_skip
   float voxel_weight = volume_weight_matrix[VOLUME_SLICE_IDX(y,x)];
#endif
   
   unsigned int i;
   unsigned int start_z_vox = *chunk_index * rt_chunk_size;
   unsigned int end_z_vox = start_z_vox + rt_chunk_size;

   float local_transform_matrix[TRANSFORM_MATRIX_SIZE];

   for (i=0; i<TRANSFORM_MATRIX_SIZE; i++) {
      local_transform_matrix[i] = transform_matrix[i];
   }

   float combined[3];
   combined[0] = combined_matrix[VOLUME_SLICE_IDX(y,x)];
   combined[1] = combined_matrix[VOLUME_SLICE_IDX(y,x) + Z_SLICE_SIZE];

   // Loop over the slices to reconstruct

   unsigned int voxel_index = VOLUME_SLICE_IDX(y,x);
   unsigned int reg_proj_row_offset = *proj_row_offset;
   
   for (i=start_z_vox; i<end_z_vox; i++) {
      combined[2] = shared_z_voxel_coordinates[i];
      backproject_slice(&recon_voxel,
			&local_transform_matrix[0],
			&combined[0],
			proj_data, 
			&reg_proj_row_offset);

#ifndef volume_weight_skip
      recon_voxel *= voxel_weight;
#endif

      recon_chunk[voxel_index] += recon_voxel;

      voxel_index += Z_SLICE_SIZE;
   }
}


