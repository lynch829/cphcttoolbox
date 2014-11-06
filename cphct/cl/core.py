#!/usr/bin/python
# -*- coding: utf-8 -*-

#
# --- BEGIN_HEADER ---
#
# core - OpenCL specific core helpers
# Copyright (C) 2011-2014  The Cph CT Toolbox Project lead by Brian Vinter
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

"""OpenCL specific core helper functions"""

from cphct.io import create_path_dir
from cphct.cl import opencl, RuntimeError as clRuntimeError, gpuarray
from cphct.log import logging
from cphct.npycore import sqrt
from cphct.misc import nextpow2


def __get_available_gpu_devices(conf):
    """Return a handle to the available GPU devices.

    Parameters
    ----------
    conf : dict
        Configuration dictionary.

    Returns
    -------
    output : list
        Returns a list of available GPU devices.
    """

    gpu_module = conf['gpu']['module']
    platform_list = gpu_module.get_platforms()
    gpu_devices = []
    for platform in platform_list:
        try:
            gpu_devices += \
                platform.get_devices(gpu_module.device_type.GPU)
        except clRuntimeError:

            # Probably found a CPU platform

            pass
    return gpu_devices


def __get_gpu_device(gpu_context):
    """
    Extract GPU device handle from initialized OpenCL contexts
    
    Parameters
    ----------
    gpu_context : Object
        Initialized OpenCL context
      
    Returns
    -------
    output : Object
       Returns gpu devices object
    """

    return gpu_context.devices[0]


def __get_gpu_specs(gpu_context):
    """
    Extract dictionary of GPU specs from initialized OpenCL gpu contexts
    
    Parameters
    ----------
    gpu_context : Object
        Initialized OpenCL context

    Returns
    -------
    output : dict
       Returns gpu device specs dictionary
    """

    dev = __get_gpu_device(gpu_context)
    specs = {}
    for i in dir(dev):
        if not hasattr(dev, i) or i.startswith('_'):
            continue
        key = str(i).upper()
        val = getattr(dev, i)

        # We only want integer specs for the rt vars

        if isinstance(val, int):
            specs[key] = val
        elif key == 'MAX_WORK_ITEM_SIZES':
            specs['MAX_WORK_ITEM_SIZE_X'] = int(val[0])
            specs['MAX_WORK_ITEM_SIZE_Y'] = int(val[1])
            specs['MAX_WORK_ITEM_SIZE_Z'] = int(val[2])

    return specs


def __check_gpu_environment(gpu_specs):
    """Check if the GPU environment is usable.

    Parameters
    ----------
    gpu_specs : dict
       Gpu device specs dictionary

    Returns
    -------
    output : dict
       True if GPU environment is usable.

    Raises
    ------
    pyopencl.RuntimeError:
       If GPU environment lack 3D grid support
    """

    if not 'MAX_WORK_ITEM_SIZE_Z' in gpu_specs \
        or gpu_specs['MAX_WORK_ITEM_SIZE_Z'] < 1:
        raise clRuntimeError('GPU environment lack 3D grid support')

    return True


def __switch_active_gpu(conf, gpu_id):
    """
    This activates the GPU with *gpu_id* deactivating the previous active GPU if any.
    Because the OpenCL driver only allows a CPU process to communicate with a single
    GPU at a time, this is necessary if using multiple GPUs from a single CPU processes.
    
    Parameters
    ----------
    conf : dict
        Configuration dictionary.
    gpu_id : int
        GPU device index
    Returns
    -------
    output : dict
        Returns configuration dictionary with active GPU set 
    """

    if 'active_id' in conf['gpu'] and conf['gpu']['active_id'] \
        == gpu_id:
        pass
    else:

        # Deactive current active GPU if set

        old_active_context = get_active_gpu_context(conf)
        if not old_active_context is None:
            conf['gpu']['queue'].finish()
            del conf['gpu']['barrier']
            del conf['gpu']['context'][conf['gpu']['active_id']]
            del old_active_context

        # Set new active gpu id

        conf['gpu']['active_id'] = gpu_id
        new_active_context = get_active_gpu_context(conf)

        # Activate context

        gpu_module = conf['gpu']['module']
        conf['gpu']['queue'] = \
            gpu_module.CommandQueue(new_active_context)

        # Expose sync function for e.g. timelog to use independently of engine

        conf['gpu']['barrier'] = lambda cfg: gpu_barrier(cfg)

    return conf


def get_active_gpu_context(conf):
    """
    Returns the context of the active GPU device
    
    Parameters
    ----------
    gpu_context : Object
        Initialized OpenCL context

    Returns
    -------
    output : Object
        Returns the active OpenCL context, 
        if not set *None* is returned
    """

    gpu_context = None

    if 'active_id' in conf['gpu']:
        gpu_id = conf['gpu']['active_id']
        gpu_context = conf['gpu']['context'][gpu_id]

    return gpu_context


def gpu_init_mod(conf):
    """Initialize GPU access module. Sets gpu_module for further use.

    Parameters
    ----------
    conf : dict
        Configuration dictionary.

    Returns
    -------
    output : dict
        Returns configuration dictionary with GPU module handle inserted.
    """

    conf['gpu']['module'] = opencl
    return conf


def gpu_device_count(conf):
    """Count the available GPU devices.

    Parameters
    ----------
    conf : dict
        Configuration dictionary.

    Returns
    -------
    output : int
        Returns the number of GPU devices.
    """

    return len(__get_available_gpu_devices(conf))


def gpu_mem_info(conf):
    """Fetch the available and used GPU memory.

    Parameters
    ----------
    conf : dict
        Configuration dictionary.

    Returns
    -------
    output : tuple
        Returns a tuple of free and total GPU memory in bytes.
    """

    gpu_ctx = get_active_gpu_context(conf)
    gpu_device = __get_gpu_device(gpu_ctx)

    # OpenCL does not support querying of actaully used GPU mem

    return (-42, gpu_device.global_mem_size)


def gpu_init_ctx(conf):
    """Initialize GPU context access. Sets gpu_context handle for further use.

    Parameters
    ----------
    conf : dict
        Configuration dictionary.

    Returns
    -------
    output : dict
        Returns configuration dictionary with GPU context handle inserted.
    """

    gpu_module = conf['gpu']['module']
    gpu_device_index = conf['gpu_device_index']
    gpu_devices = __get_available_gpu_devices(conf)
    conf['gpu']['context'] = {}
    conf['gpu']['context'][gpu_device_index] = \
        gpu_module.Context(devices=[gpu_devices[gpu_device_index]])
    __switch_active_gpu(conf, gpu_device_index)

    # We expect that all initialized GPU devices are the same
    # NOTE: When making support for multiple GPUs we must
    #       introduce a check for this here

    conf['gpu']['specs'] = __get_gpu_specs(get_active_gpu_context(conf))

    __check_gpu_environment(conf['gpu']['specs'])

    return conf


def gpu_barrier(conf):
    """Forces GPU barrier based on conf settings
    
    Parameters
    ----------
    conf : dict
       Dictionary with configuration values.
    """

    gpu_module = conf['gpu']['module']
    queue = conf['gpu']['queue']
    gpu_module.enqueue_barrier(queue).wait()


def gpu_exit(conf):
    """Clean up after use of GPU. Clears gpu_module and gpu_context handles
    from conf and they should not be used after this call.

    Parameters
    ----------
    conf : dict
        Configuration dictionary.

    Returns
    -------
    output : dict
        Returns configuration dictionary with GPU handles cleared.
    """

    if conf['gpu'].get('queue', None):
        conf['gpu']['queue'].finish()
        del conf['gpu']['queue']

    if 'active_id' in conf['gpu']:
        logging.debug('De-activating GPU: %s' % conf['gpu']['active_id'
                      ])
        del conf['gpu']['active_id']

    if 'context' in conf['gpu']:
        for gpu_id in conf['gpu']['context'].keys():
            logging.debug('Detaching context for GPU: %s' % gpu_id)
            del conf['gpu']['context'][gpu_id]
        del conf['gpu']['context']

    if 'specs' in conf['gpu']:
        del conf['gpu']['specs']

    if 'module' in conf['gpu']:
        del conf['gpu']['module']

    return conf


def log_gpu_specs(conf):
    """Simple helper to write out GPU specs using logging

    Parameters
    ----------
    conf : dict
        Configuration dictionary.
    """

    logging.debug('***************** GPU Specifications *****************'
                  )
    for gpu_id in conf['gpu']['context']:
        gpu_context = conf['gpu']['context'][gpu_id]
        gpu_device = __get_gpu_device(gpu_context)
        logging.info('GPU %d: %s' % (gpu_id, gpu_device.name))

    gpu_attrs = conf['gpu']['specs']
    for (key, val) in gpu_attrs.items():
        logging.debug('%s: %s' % (key, val))
    logging.debug('******************************************************'
                  )


def replace_constants(text, remap, wrap=False):
    """Replace keys from remap dictionary with values in text string. We sort
    and reverse the remap keys to make sure any prefix keys get handled last.
    All values are wrapped in parenthesis to avoid macro breakage, etc if wrap
    is set.

    Parameters
    ----------
    text : str
        String of text to replace constants in.
    remap : dict
        Remap dictionary used to replace all occurrences of key with value.
    wrap : bool
        If the replace variables should be wrapped in parentheses for safety.

    Returns
    -------
    output : str
        Returns text with all words in remap keys replaced by the mapped value.
    """

    res = text
    remap_tuples = [i for i in remap]
    remap_tuples.sort()
    remap_tuples.reverse()
    if wrap:
        pattern = '(%s)'
    else:
        pattern = '%s'
    for (key, val) in remap_tuples:
        res = res.replace(key, pattern % val)
    return res


def load_kernels(kernels_load_path, use_binary=False):
    """Load OpenCL kernels

    Parameters
    ----------
    kernels_load_path : str
       Path to OpenCL source/binary
    use_binary : bool
       Open file as raw binary
    
    Returns
    -------
    output : str
       Returns the loaded OpenCL kernels in source or binary format
    """

    kernels_code = ''
    open_flags = 'r'
    if use_binary:
        open_flags += 'b'
    try:
        kernels_fd = open(kernels_load_path, open_flags)
        kernels_code += kernels_fd.read()
        kernels_fd.close()
    except Exception, exc:
        logging.error('failed to load OpenCL kernels: %s' % exc)

    return kernels_code


def load_kernels_clbin(kernel_binary_path):
    """Load precompiled OpenCL kernels binary
    
    Parameters
    ----------
    kernel_binary_path : str
       Path to OpenCL kernels binary
    
    Returns
    -------
    output : str
       Returns the loaded OpenCL kernels in binary format
    """

    return load_kernels(kernel_binary_path, True)


def load_kernels_source(kernels_code_path):
    """Load OpenCL kernels source
    
    Parameters
    ----------
    kernels_code_path : str
       Path to OpenCL kernels source
    
    Returns
    -------
    output : str
       Returns the loaded OpenCL kernels in source format
    """

    return load_kernels(kernels_code_path, False)


def generate_gpu_init(conf, rt_const):
    """Define GPU helpers and inline variables 
    that are constant at kernel runtime.

    Parameters
    ----------
    rt_const : dict
        Dictionary with list of integer, float and string  
        variable names to inline as constants.

    Returns
    -------
    output : str
        Returns OpenCL kernel runtime configuration based on *rt_const* entries
        based on *rt_const* entries        
    """

    output = \
        '''
/* --- BEGIN OPENCL HELPERS --- */

/* Thread ID helpers
CUDA vs OpenCL:
 - blocks in grid are equivalent to groups
 - threads in block is equivalent to local
*/

#define THREAD_ID_X (get_local_id(0))
#define THREAD_ID_Y (get_local_id(1))
#define THREAD_ID_Z (get_local_id(2))
#define BLOCK_DIM_X (get_local_size(0))
#define BLOCK_DIM_Y (get_local_size(1))
#define BLOCK_DIM_Z (get_local_size(2))
#define BLOCK_ID_X (get_group_id(0))
#define BLOCK_ID_Y (get_group_id(1))
#define BLOCK_ID_Z (get_group_id(2))

#define GET_GLOBAL_ID_X (get_global_id(0))
#define GET_GLOBAL_ID_Y (get_global_id(1))
#define GET_GLOBAL_ID_Z (get_global_id(2))

#define GET_LOCAL_ID_X (get_local_id(0))
#define GET_LOCAL_ID_Y (get_local_id(1))
#define GET_LOCAL_ID_Z (get_local_id(2))

/* Keyword helpers 
CUDA vs OpenCL:
 - kernels are prefixed with __global__ vs __kernel 
 - global memory args are prefixed with '' vs __global
 - block shared mem is marked with __shared__ vs __local
 - thread private mem is marked with '' vs __private
 - functions are forced inline with __forceinline__ vs __inline
 - device functions are prefixed with __device__ vs ''
*/

#define KERNEL __kernel
#define GLOBALMEM __global
#define SHAREDMEM __local
#define PRIVATEMEM __private
#define FORCEINLINE __inline
#define DEVICE 


/* Thread memory syncronization within blocks:
CUDA vs OpenCL:
 - sync is issued with __syncthreads() vs barrier(flags)
please read about the meaning of flags for correctness
*/

#define sync_shared_mem() barrier(CLK_LOCAL_MEM_FENCE)

/* Macros fixing OpenCL quirks */

// use conversion helpers in OpenCL

#define int(x) (convert_int(x))
#define float(x) (convert_float(x))

/* --- END OPENCL HELPERS */

/* --- BEGIN AUTOMATIC RUNTIME CONFIGURATION --- */

'''
    for name in conf['gpu']['specs']:
        output += '''#define rt_gpu_specs_%s (%d)
''' % (name,
                conf['gpu']['specs'][name])

    for name in rt_const['int']:
        output += '''#define rt_%s ((int)%d)
''' % (name, conf[name])

    for name in rt_const['float']:
        output += '''#define rt_%s ((float)%.32f)
''' % (name,
                conf[name])

    for name in rt_const['str']:
        if conf[name]:
            output += \
                '''#define rt_%s_%s ((unsigned long long)0x%s)
''' \
                % (name, conf[name], conf[name].encode('hex'))
            output += '''#define rt_%s rt_%s_%s
''' % (name, name,
                    conf[name])
    output += '''
/* --- END AUTOMATIC RUNTIME CONFIGURATION --- */
'''
    return output


def compile_kernels(conf, kernels_code, precompiled=False):
    """Compile OpenCL kernels from raw or precompiled *kernels_code*

    Parameters
    ----------
    conf : dict
       Configuration dictionary.
    kernels_code : str
       OpenCL kernel source or precompiled code to fully compile
        
    Returns
    -------
    output : tuple
       Returns (const_code, cl_kernels, clbin_data) with types
       (str, pyopencl.Program, str).

    """

    kernels = None
    clbin_data = None
    const_code = None
    gpu_module = conf['gpu']['module']
    gpu_ctx = get_active_gpu_context(conf)
    gpu_device = __get_gpu_device(gpu_ctx)

    if precompiled:
        source_code = const_code = ''
    else:
        source_code = kernels_code

    try:

        # Escape any percent chars in the raw code before conf expansion

        const_code = replace_constants(source_code, [('%', '%%')],
                wrap=False)
        const_code = replace_constants(const_code,
                conf['host_params_remap'])
        const_code = const_code % conf

        # Revert any percent chars supposed to be left there

        const_code = replace_constants(const_code, [('%%', '%')],
                wrap=False)

        # OpenCL compile flags are described at:
        # https://www.khronos.org/registry/cl/sdk/1.0/docs/man/xhtml/clBuildProgram.html

        cc_options = ['-cl-mad-enable', '-cl-fast-relaxed-math']

        if precompiled:
            program = gpu_module.Program(get_active_gpu_context(conf),
                    [gpu_device], [kernels_code])
        else:
            program = gpu_module.Program(get_active_gpu_context(conf),
                    const_code)
        kernels = program.build(options=cc_options)
        clbin_data = program.binaries[0]
    except Exception, exc:
        logging.error('compile compute kernels failed: %s' % exc)
        logging.debug(const_code)

    return (const_code, kernels, clbin_data)


def gpu_kernels_auto_init(conf, rt_const):
    """Prepare OpenCL kernels based on conf settings and *rt_const* entries
    
    Parameters
    ----------
    conf : dict
       Configuration dictionary.
    rt_const : dict
        Dictionary with list of integer, float and string  
        variable names to inline as constants.

    Returns
    -------
    output : dict
        Configuration dictionary with updated *const_code*, *clbin_data* and *cl_kernels*
    
    """

    conf['const_code'] = None
    conf['clbin_data'] = None
    conf['cl_kernels'] = None

    if conf['load_gpu_binary_path']:
        kernel_binary = load_kernels_clbin(conf['load_gpu_binary_path'])
        (conf['const_code'], conf['cl_kernels'], conf['clbin_data']) = \
            compile_kernels(conf, kernel_binary, True)
    elif conf['load_gpu_kernels_path']:
        kernels_code = ''
        if conf['load_gpu_init_path']:
            kernels_code += \
                load_kernels_source(conf['load_gpu_init_path'])
        else:
            kernels_code += generate_gpu_init(conf, rt_const)

        kernels_code += load_kernels_source(conf['load_gpu_kernels_path'
                ])

        (conf['const_code'], conf['cl_kernels'], conf['clbin_data']) = \
            compile_kernels(conf, kernels_code, False)


def gpu_save_kernels(conf):
    """Save compiled runtime constant and clbin code for kernels.

    Parameters
    ----------
    conf : dict
        Configuration dictionary.
    """

    if conf['save_gpu_kernels_path'] and conf['const_code']:
        create_path_dir(conf['save_gpu_kernels_path'])
        kernel_fd = open(conf['save_gpu_kernels_path'], 'w')
        kernel_fd.write(conf['const_code'])
        kernel_fd.close()
    if conf['save_gpu_binary_path'] and conf['clbin_data']:
        create_path_dir(conf['save_gpu_binary_path'])
        kernel_fd = open(conf['save_gpu_binary_path'], 'w')
        kernel_fd.write(conf['clbin_data'])
        kernel_fd.close()


def gpu_array_from_alloc(conf, mem_chunk, shape, data_type):
    """Wrap raw mem_alloc in a GPUArray object with given shape and data_type
    for easy use e.g. in plugins.

    Parameters
    ----------
    conf : dict
        Configuration dictionary.
    mem_chunk : DeviceAllocation
        A previously allocated linear chunk of device memory.
    shape : tuple
        The shape of the resulting GPUArray.
    data_type : dtype
        The data type of the resulting GPUArray.

    Returns
    -------
    output : GPUArray
        Returns a GPUArray wrapping the existing allocation.
    """

    return gpuarray.GPUArray(conf['gpu']['queue'], shape, data_type,
                             data=mem_chunk)


def gpu_alloc_from_array(gpuarray_obj):
    """Extract raw memory allocation from GPUArray object to e.g. revert
    temporary GPUArray wrapping after use in plugins.

    Parameters
    ----------
    gpuarray_obj : GPUArray
        A previously GPUArray wrapped linear chunk of device memory.

    Returns
    -------
    output : DeviceAllocation
        Returns the underlying raw memory allocation for gpuarray_obj.
    """

    mem_chunk = gpuarray_obj.data
    return mem_chunk


def gpu_array_alloc_offset(conf, gpuarray_obj, offset, shape):
    """Offsets the raw memory allocation associated with 
    *gpuarray_obj* by *offset* elements.

    Parameters
    ----------
    conf : dict
        Configuration dictionary.
    gpuarray_obj : GPUArray
        A previously GPUArray wrapped linear chunk of device memory.
    offset : int
        Offset GPUArray elements in the flat memory space.
    shape : tuple
        Shape of the offset GPUArray.
        
    Returns
    -------
    output : GPUArray
        Returns *gpuarray_obj* offset by *byteoffset*
    """

    offset_bytes = gpuarray_obj.dtype.itemsize * offset
    nr_elms = reduce(lambda x, y: x * y, shape)
    size_bytes = nr_elms * gpuarray_obj.dtype.itemsize

    offset_gpuarray_alloc = \
        gpu_alloc_from_array(gpuarray_obj).get_sub_region(offset_bytes,
            size_bytes)
    offset_gpuarray_obj = gpu_array_from_alloc(conf,
            offset_gpuarray_alloc, shape, gpuarray_obj.dtype)

    return offset_gpuarray_obj


def get_gpu_layout(
    chunks,
    rows,
    cols,
    max_gpu_threads_pr_block,
    ):
    """
    Get GPU block layout based on rows and cols,
    and the number of threads pr. block. We aim at square layouts.
    We always return a grid z-dimension of 1 to support e.g. NVidia devices
    with compute capability less than 2.0 and CUDA versions prior to 4.0.
   
    Parameters
    ----------
    chunks : int
       The number of chunks in the GPU grid layout
    rows : int
       The number of rows in the GPU grid layout
    cols : int
       The number of columns in the GPU grid layout
    max_gpu_threads_pr_block : int
       Maximum number of GPU threads per block. Used to fit specific hardware.

    Returns
    -------
    output : tuple
       Tuple of GPU block and grid dimension tuples on the form:
       ((block_xdim, block_ydim, block_zdim), (grid_xdim, grid_ydim, grid_zdim))
            
    Raises
    ------
    ValueError:
       If unable to generate a valid layout from the given parameters
    """

    rows = float(rows)
    cols = float(cols)

    gpu_thread_pr_block = float(max_gpu_threads_pr_block)

    # Calculate block x and y dim based on the image width, height.

    # Make sure we don't exceed the max number of threads in any block
    # dimension even for severely skewed layouts like 1x512 matrices

    block_xdim = nextpow2(int(round(sqrt(gpu_thread_pr_block / (rows
                          / cols)))))

    while block_xdim > gpu_thread_pr_block or cols % block_xdim != 0 \
        and block_xdim > 1.0:
        block_xdim /= 2

    if block_xdim < 1.0:
        raise ValueError('Wrong block_xdim: %s' % block_xdim)

    grid_xdim = cols / block_xdim
    if grid_xdim < 1.0 or cols % block_xdim != 0:
        raise ValueError('Wrong grid_xdim: %s' % grid_xdim)

    block_ydim = gpu_thread_pr_block / block_xdim
    while rows % block_ydim != 0 and block_ydim > 1.0:
        block_ydim /= 2

    if block_ydim < 1.0:
        raise ValueError('Wrong block_ydim: %s' % block_ydim)

    grid_ydim = rows / block_ydim
    if grid_ydim < 1.0 or rows % block_ydim != 0:
        raise ValueError('Wrong grid_ydim: %s' % grid_ydim)

    # In cl grid is global thread layout so we don't divide by block dim

    return ((int(block_xdim), int(block_ydim), 1), (int(cols),
            int(rows), int(chunks)))
