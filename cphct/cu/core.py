#!/usr/bin/python
# -*- coding: utf-8 -*-

#
# --- BEGIN_HEADER ---
#
# core - cuda specific core helpers
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

"""CUDA specific core helper functions"""

import os

from cphct.io import create_path_dir
from cphct.cu import driver as cuda, compiler, gpuarray
from cphct.log import logging
from cphct.npycore import sqrt
from cphct.misc import nextpow2


def __get_gpu_device(gpu_context):
    """
    Extract GPU device handles from initialized CUDA contexts
    
    Parameters
    ----------
    gpu_context : Object
        Initialized CUDA context
      
    Returns
    -------
    output : Object
       Returns gpu devices object
    """

    return gpu_context.get_device()


def __get_gpu_specs(gpu_context):
    """
    Extract dictionary of GPU specs from initialized CUDA gpu contexts
    
    Parameters
    ----------
    gpu_context : Object
        Initialized CUDA context

    Returns
    -------
    output : dict
       Returns gpu devices specs dictionary
    """

    specs = __get_gpu_device(gpu_context).get_attributes()
    return dict([(str(i), j) for (i, j) in specs.items()])


def __switch_active_gpu(conf, gpu_id):
    """
    This activates the GPU with *gpu_id* deactivating the previous active GPU if any.
    Because the CUDA driver only allows a CPU process to communicate with a single
    GPU at a time, this is necessary if using multiple GPUs from a single CPU processes.
    
    Parameters
    ----------
    conf : dict
        Configuration dictionary.

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

        old_active_context = __get_active_gpu_context(conf)
        if not old_active_context is None:
            old_active_context.pop()

        # Set new active gpu id

        conf['gpu']['active_id'] = gpu_id
        new_active_context = __get_active_gpu_context(conf)

        # Activate context

        new_active_context.push()

    return conf


def __get_active_gpu_context(conf):
    """
    Returns the context of the active GPU device
    
    Parameters
    ----------
    gpu_context : Object
        Initialized CUDA context

    Returns
    -------
    output : Object
        Returns the active CUDA context, 
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

    conf['gpu']['module'] = cuda
    conf['gpu']['module'].init()
    return conf


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
    cuda_device_index = conf['cuda_device_index']

    # Use tempdir for the CUDA compute cache documented in CUDA Toolkit Docs

    os.environ["CUDA_CACHE_PATH"] = os.path.join(conf["temporary_directory"],
                                                 '.nv', 'ComputeCache')

    conf['gpu']['context'] = {}
    conf['gpu']['context'][cuda_device_index] = \
        gpu_module.Device(cuda_device_index).make_context()

    __switch_active_gpu(conf, cuda_device_index)

    # We expect that all initialized GPU devices are the same
    # NOTE: When making support for multiple GPUs we must
    #       introduce a check for this here

    conf['gpu']['specs'] = \
        __get_gpu_specs(__get_active_gpu_context(conf))

    return conf


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

    if 'active_id' in conf['gpu']:
        logging.debug('De-activating GPU: %s' % conf['gpu']['active_id'
                      ])
        __get_active_gpu_context(conf).pop()
        del conf['gpu']['active_id']

    if 'context' in conf['gpu']:
        for gpu_id in conf['gpu']['context']:
            logging.debug('Detaching context for GPU: %s' % gpu_id)
            conf['gpu']['context'][gpu_id].detach()

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
        logging.info('GPU %d: %s' % (gpu_id, gpu_device.name()))

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


def load_kernels_cubin(cubin_load_path):
    """Load kernel precompiled cubin kernel.

    Parameters
    ----------
    cubin_load_path : str
       Filepath to cubin kernel

    Returns
    -------
    output : object
        Returns the loaded CUDA kernels
    """

    output = None

    try:
        output = cuda.module_from_file(cubin_load_path)
    except Exception, exc:
        logging.error('faild to load precompiled module: %s' % exc)

    return output


def load_kernel_source(kernel_code_filepath):
    """Load CUDA kernel source
    
    Parameters
    ----------
    cubin_load_path : str
       Filepath to CUDA source
    
    Returns
    -------
    output : str
       Returns the loaded CUDA source
    """

    kernel_code = ''

    try:
        kernel_code_fd = open(kernel_code_filepath, 'r')
        kernel_code += kernel_code_fd.read()
        kernel_code_fd.close()
    except Exception, exc:

        logging.error('faild to load CUDA kernel source: %s' % exc)

    return kernel_code


def generate_gpu_init(conf, rt_const):
    """Prepare GPU inlining of variables that are constant at kernel runtime.

    Parameters
    ----------
    rt_const : dict
        Dictionary with list of integer, float and string  
        variable names to inline as constants.

    Returns
    -------
    output : str
        Returns CUDA kernel runtime configuration based on *rt_const* entries
    """

    output = \
        '''
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


def compile_kernels(conf, kernel_code):
    """Compile CUDA kernels from *kernel_code*

    Parameters
    ----------
    conf : dict
       Configuration dictionary.
    kernel_source : str
       CUDA kernel source to compile
        
    Returns
    -------
    output : tuple
       Returns (const_code, cu_kernels, cubin_data) with types
       (str, pycuda.compiler.SourceModule, str).

    """

    kernels = None
    cubin_data = None
    const_code = None

    try:

        # Escape any percent chars in the raw code before conf expansion

        const_code = replace_constants(kernel_code, [('%', '%%')],
                wrap=False)
        const_code = replace_constants(const_code,
                conf['host_params_remap'])
        const_code = const_code % conf

        # Revert any percent chars supposed to be left there

        const_code = replace_constants(const_code, [('%%', '%')],
                wrap=False)

        nvcc_options = ['--compiler-options', '-fno-strict-aliasing',
                        '--use_fast_math', '-DUNIX', '-O2']

        # nvcc_options = []
        # arch_opts = ["compute_20"]

        arch_opts = None

        # code_opts = ["compute_20"]

        code_opts = None

        kernels = compiler.SourceModule(const_code,
                options=nvcc_options, arch=arch_opts, code=code_opts)
        cubin_data = compiler.compile(const_code, options=nvcc_options,
                arch=arch_opts, code=code_opts)
    except Exception, exc:

        logging.error('load compute kernels failed: %s' % exc)
        logging.debug(const_code)

    return (const_code, kernels, cubin_data)


def gpu_kernel_auto_init(conf, rt_const):
    """Prepare CUDA kernels based on conf settings and *rt_const* entries
    
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
        Configuration dictionary with updated *const_code*, *cubin_data* and *cu_kernels*
    
    """

    conf['const_code'] = None
    conf['cubin_data'] = None
    conf['cu_kernels'] = None

    if conf['load_gpu_binary_path']:
        conf['cu_kernels'] = \
            load_kernels_cubin(conf['load_gpu_binary_path'])
    elif conf['load_gpu_kernels_path']:

        kernel_code = ''
        if conf['load_gpu_init_path']:
            gpu_init_fd = open(conf['load_gpu_init_path'], 'r')
            kernel_code += gpu_init_fd.read()
            gpu_init_fd.close()
        else:
            kernel_code += generate_gpu_init(conf, rt_const)

        kernel_code += load_kernel_source(conf['load_gpu_kernels_path'])

        (conf['const_code'], conf['cu_kernels'], conf['cubin_data']) = \
            compile_kernels(conf, kernel_code)


def gpu_save_kernels(conf):
    """Save compiled runtime constant and cubin code for kernels.

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
    if conf['save_gpu_binary_path'] and conf['cubin_data']:
        create_path_dir(conf['save_gpu_binary_path'])
        kernel_fd = open(conf['save_gpu_binary_path'], 'w')
        kernel_fd.write(conf['cubin_data'])
        kernel_fd.close()


def gpu_array_from_alloc(mem_chunk, shape, data_type):
    """Wrap raw mem_alloc in a GPUArray object with given shape and data_type
    for easy use e.g. in plugins.

    Parameters
    ----------
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

    return gpuarray.GPUArray(shape, data_type, allocator=lambda x: \
                             mem_chunk)


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

    mem_chunk = gpuarray_obj.gpudata
    return mem_chunk


def gpu_pointer_from_array(gpuarray_obj):
    """Extract memory pointer to raw memory allocation 
    from GPUArray object to e.g. offset GPU memory.

    Parameters
    ----------
    gpuarray_obj : GPUArray
        A previously GPUArray wrapped linear chunk of device memory.

    Returns
    -------
    output : int
        Returns the address of the underlying raw memory allocation 
        for gpuarray_obj.
    """

    pointer = gpuarray_obj.ptr
    return pointer


def gpu_array_alloc_offset(gpuarray_obj, offset, shape):
    """Offsets the raw memory allocation associated with 
    *gpuarray_obj* by *offset* elements.

    Parameters
    ----------
    gpuarray_obj : GPUArray
        A previously GPUArray wrapped linear chunk of device memory.
    offset : int
        Offset GPUAarray elements in the flat memory space.
    shape : tuple
        Shape of the offsat GPUArray.
        
    Returns
    -------
    output : GPUArray
        Returns *gpuarray_obj* offsat by *byteoffset*
    """

    byteoffset = offset * gpuarray_obj.dtype.itemsize
    gpuarray_obj_offsat_ptr = gpu_pointer_from_array(gpuarray_obj) \
        + byteoffset
    offsat_gpuarray_obj = gpu_array_from_alloc(gpuarray_obj_offsat_ptr,
            shape, gpuarray_obj.dtype)

    return offsat_gpuarray_obj


def get_gpu_layout(rows, cols, max_gpu_threads_pr_block):
    """
    Get GPU block layout based on rows and cols,
    and the number of threads pr. block. We aim at square layouts.
    We always return a grid z-dimension of 1 to support devices with
    compute capability less than 2.0 and CUDA versions prior to 4.0.
   
    Parameters
    ----------
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
       ((block_xdim, block_ydim, block_zdim), (grid_xdim, grid_ydim, 1))
            
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

    return ((int(block_xdim), int(block_ydim), 1), (int(grid_xdim),
            int(grid_ydim), 1))


