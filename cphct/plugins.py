#!/usr/bin/python
# -*- coding: utf-8 -*-

#
# --- BEGIN_HEADER ---
#
# plugins - application plugin framework
# Copyright (C) 2012-2013  The Cph CT Toolbox Project lead by Brian Vinter
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

"""Plugin framework for optional application extensions that may be provided
by users.

All plugin modules should implement the mandatory init/exit functions:
plugin_init(conf, ..)
plugin_exit(conf, ..)

that will automatically be called once with the actual plugin arguments before
and after all plugin use respectively. They may be used to set up and tear
down plugin data structures but can just return without any actions otherwise.

The actual plugin actions are implemented in the four functions:
load_input(input_data, input_meta, conf, ..)
preprocess_input(input_data, input_meta, conf, ..)
postprocess_output(output_data, output_meta, conf, ..)
save_output(output_data, output_meta, conf, ..)

that take at least an array, a meta data list and a configuration dictionary
as input, but where '..' should include any other valid positional or named
arguments that may be passed to the plugin. They should modify the array
inline if possible and always return the resulting array.

Please note that the plugin functions will typically be called on a chunk of
the complete input and output. Thus the functions should be flexible enough to
handle variable chunks, or detect and fail if chunk is incompatible with the
processing.
"""

import os
import sys
import traceback

allowed_plugin_hooks = ['load_input', 'preprocess_input',
                        'postprocess_output', 'save_output']
internal_plugin_hooks = ['plugin_init', 'plugin_exit']


def unsupported_handler(*args, **kwargs):
    """Shared helper to mark plugin unsuitable for particular operations.

    Parameters
    ----------
    *args : positional arguments
        Any positional arguments.
    **kwargs : keyword arguments
        Any keyword arguments.

    Raises
    -------
    ValueError :
        Always as operation doesn't make sense.
    """

    raise ValueError('unsupported operation for plugin!')


def add_unsupported(plugin_mod, handler):
    """Adds the *handler* function for any targets from allowed_plugin_hooks
    that *plugin_mod* does not implement itself. This is in order to allow
    plugins to only explicitly implement relevant handlers yet still gracefully
    handle cases where a user tries to execute a plugin in an unhandled
    context.

    Parameters
    ----------
    plugin_mod : module object
        A plugin module previously loaded.
    handler : function
        A function to assign to targets that *plugin_mod* does not implement.
    """

    for target in allowed_plugin_hooks:
        if not hasattr(plugin_mod, target):
            setattr(plugin_mod, target, handler)


def plugin_base_dirs(conf):
    """Return a list of base plugin search directories.
    Plugins are picked up from sub directories of the:
     * global toolbox installation path (/path/to/cphcttoolbox)
     * toolbox dot directory in the user home (~/.cphcttoolbox)
     * current directory (./)
    and in that order.

    Parameters
    ----------
    conf : dict
        Configuration dictionary.
        
    Returns
    -------
    output : list of str
        Returns a list of plugin search directory paths
    """

    # Parent dir of this module dir is toolbox base

    global_base = conf['cphcttoolbox_base']
    user_base = os.path.expanduser(os.path.join('~', '.cphcttoolbox'))
    local_base = os.path.abspath('.')
    return [global_base, user_base, local_base]


def app_plugin_dirs(app_names, engine_dir, conf):
    """Return a list of application-specific plugin search directories for the
    application registering with the names in the *app_names* list and the
    given *engine_dir*.
    The list contains plugin directory paths sorted in growing priority order.

    Parameters
    ----------
    app_names : list of str
        List of application names.
    engine_dir : str
        Back end calculation engine sub directory name.
    conf : dict
        Configuration dictionary.
        
    Returns
    -------
    output : list of str
        Returns a list of engine-specific plugin directories for the given
        application.
    """

    plugin_paths = []
    plugin_dirs = plugin_base_dirs(conf)

    # Search and add plugin directories in increasing priority order

    plugin_prefixes = ['cphct', ''] + app_names
    for base_path in plugin_dirs:
        for plugin_dir in ['%splugins' % pre for pre in plugin_prefixes]:
            dir_path = os.path.join(base_path, plugin_dir, engine_dir)
            plugin_paths.append(dir_path)
    return plugin_paths


def app_plugin_paths(app_names, engine_dir, conf):
    """Return a list of available plugins for the application registering with
    the names in the *app_names* list and the given *engine_dir*.
    The list contains tuples of plugin location and name and it is sorted in
    growing priority order in case a plugin name appears more than once.
    Plugins are picked up from the:
     * global toolbox installation path (/path/to/cphcttoolbox)
     * toolbox dot directory in the user home (~/.cphcttoolbox)
     * current directory (./)
    and in that order.
    For each search location any 'cphctplugins', 'plugins' and 'Xplugins'
    directories (where X is a name in *app_names*) will be searched for python
    modules in the *engine_dir* sub directories. I.e. cphctplugins/npy will be
    searched for numpy plugins.

    Parameters
    ----------
    app_names : list of str
        List of application names.
    engine_dir : str
        Back end calculation engine sub directory name.
    conf : dict
        Configuration dictionary.
        
    Returns
    -------
    output : list of tuple
        Returns a list of plugin directory and name tuples
    """

    plugin_paths = []
    plugin_dirs = app_plugin_dirs(app_names, engine_dir, conf)

    # Search and add plugins in increasing priority order

    for dir_path in plugin_dirs:
        if os.path.isdir(dir_path):
            dir_files = os.listdir(dir_path)
            for file_name in dir_files:
                (mod_name, ext) = os.path.splitext(file_name)
                mod_path = os.path.join(dir_path, mod_name)
                if ext == '.py' or os.path.isdir(mod_path):
                    plugin_paths.append((dir_path, mod_name))

    return plugin_paths


def load_plugins(app_names, engine_dir, conf):
    """Load plugins specified in conf using plugin paths based on *app_names*
    and *engine_dir*. In case of multiple plugins with the same base name the
    plugin_paths list is used as growing priority order so that the last
    matching plugin is used.
    Returns a tuple containing a dictionary with loaded plugins and a
    dictionary containing any loading errors encountered. Both dictionaries
    map from names in allowed_plugin_hooks to the actual data.
    Enabled hooks automatically get their optional internal plugin hooks set
    for init and clean up. The *engine_dir* parameter is used as a prefix for
    the inserted *conf* values.
    Plugin arguments from the corresponding *conf* entries are copied so that
    all subsequent plugin actions can rely solely on the returned dictionary.

    Parameters
    ----------
    app_names : list of str
        List of application names.
    engine_dir : str
        Back end calculation engine sub directory name.
    conf : dict
        Configuration dictionary.

    Returns
    -------
    output : (dict, dict)
        Returns a 2-tuple of a plugin dictionary and load error dictionary.
    """

    (plugins, errors) = ({}, {})
    orig_sys_path = sys.path
    internal_targets = ['%s_%s' % (engine_dir, i) for i in
                        internal_plugin_hooks]
    external_targets = ['%s_%s' % (engine_dir, i) for i in
                        allowed_plugin_hooks]

    # For automatic init and clean up of enabled plugins

    for auto_target in internal_targets:
        plugins[auto_target] = []

    # Locate and load plugins in increasing priority order

    plugin_paths = app_plugin_paths(app_names, engine_dir, conf)
    for target in external_targets:
        (plugins[target], errors[target]) = ([], [])

        # Conf entry is (name, args, kwargs) tuple

        index = 0
        for (req_mod, args, kwargs) in conf.get(target, []):
            use_plugin = None

            # Search backwards from the end to apply priority

            for (dir_path, mod_name) in plugin_paths[::-1]:
                if req_mod == mod_name:
                    use_plugin = (dir_path, mod_name)
                    break

            if use_plugin:
                (plugin_dir, plugin_name) = use_plugin

                # Load plugin with plugin_dir as first source but with
                # original module path appended for external dependencies.
                # Automatically add init and exit hooks for all enabled
                # plugins. Please note that we repeat init and exit for every
                # occurrence of a plugin because it may require individual init
                # and exit for each set of arguments.
                # Immediately remove module from sys.modules after import to
                # avoid caching when loading module of same name for another
                # engine.
                # We insert the plugin call arguments from conf here for
                # complete information in returned plugins dictionary.
                # Finally we prepare the plugin instance __plugin_state__
                # dictionary for use in individual plugin executions.

                sys.path = [plugin_dir] + orig_sys_path
                try:
                    plugin_mod = __import__(plugin_name)
                    plugin_mod.__plugin_state__['target'] = target
                    plugin_mod.__plugin_state__['id'] = index
                    add_unsupported(plugin_mod, unsupported_handler)
                    del sys.modules[plugin_name]
                    plugin_tuple = (plugin_name, plugin_mod, args,
                                    kwargs)
                    plugins[target].append(plugin_tuple)
                    for auto_target in internal_targets:
                        plugins[auto_target].append(plugin_tuple)
                except Exception, exc:
                    err = 'Failed to load %s plugin from %s:\n%s' \
                        % (plugin_name, plugin_dir,
                           traceback.format_exc(exc))
                    errors[target].append((plugin_name, err))
            else:
                err = 'No such plugin "%s" in plugin directories %s' \
                    % (req_mod, ', '.join(app_plugin_dirs(app_names,
                       engine_dir, conf)))
                errors[target].append((req_mod, err))
            index += 1
    sys.path = orig_sys_path
    return (plugins, errors)


def execute_plugin(
    hook,
    name,
    plugin_mod,
    args,
    kwargs,
    ):
    """Execute matching *hook* function from *plugin_mod* plugin module with
    the provided positional *args* and named *kwargs*.

    Parameters
    ----------
    hook : str
        Name of hook function
    name : str
        Name of plugin
    plugin_mod : module
        Plugin module previously loaded and prepared.
    args : list of str
        List of arguments for plugin
    kwargs : dict
        Dictionary of keyword and value pair arguments for plugin

    Returns
    -------
    output : ndarray or None
        The processed ndarray for main hooks and None for init/exit hooks.

    Raises
    -------
    ValueError
        If plugin or hook does not match any supplied plugins.
    """

    if hook.endswith('_plugin_init'):
        return plugin_mod.plugin_init(*args, **kwargs)
    elif hook.endswith('_load_input'):
        return plugin_mod.load_input(*args, **kwargs)
    elif hook.endswith('_preprocess_input'):
        return plugin_mod.preprocess_input(*args, **kwargs)
    elif hook.endswith('_postprocess_output'):
        return plugin_mod.postprocess_output(*args, **kwargs)
    elif hook.endswith('_save_output'):
        return plugin_mod.save_output(*args, **kwargs)
    elif hook.endswith('_plugin_exit'):
        return plugin_mod.plugin_exit(*args, **kwargs)
    else:
        raise ValueError('invalid plugin hook %s for %s (%s)' % (hook,
                         name, plugin_mod))


def set_plugin_var(
    conf,
    key,
    value,
    replace=False,
    ):
    """Set plugin variable *key* to *value*.
    This is used to share variables between plugins.

    Parameters
    ----------
    conf : dict
        Configuration dictionary.
    key : str
        Variable name.
    value : object
        Variable value.
    replace : bool, optional
        If True, existing variable *key* is replaced by *value*.
        
    Raises
    -------
    ValueError
        If *replace* is False and *key* already exists
    """

    if replace or key not in conf['plugin_shared_vars']:
        conf['plugin_shared_vars'][key] = value
    else:
        msg = \
            "plugin var: '%s' already exists, use replace=True to overwrite" \
            % key
        raise ValueError(msg)


def get_plugin_var(conf, key):
    """Get plugin variable value for *key*.
    This is used to share variables between plugins.
    
    Parameters
    ----------
    conf : dict
        Configuration dictionary.
    key : str
        Variable name.
        
    Returns
    -------
    output : object or None
        Returns the value for *key* or None if *key* was not previously set
    
    """

    output = None
    if key in conf['plugin_shared_vars']:
        output = conf['plugin_shared_vars'][key]

    return output


