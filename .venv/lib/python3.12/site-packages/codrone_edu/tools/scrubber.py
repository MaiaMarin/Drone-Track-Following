import os
import sys
import importlib

def clear_user_module_cache():
    """
    Clear all user modules from sys.modules cache.

    This removes cached modules from /mnt/pfr and /mnt/blockly to ensure
    fresh imports on each code execution. This is important because module
    cache can cause stale code to be used if a file is modified between runs.

    Also handles namespace packages (no __init__.py) which have __file__ = None
    but __path__ pointing to user directories.
    """
    user_prefixes = ('/mnt/pfr', '/mnt/blockly')

    modules_to_remove = []
    for mod_name, mod in sys.modules.items():
        # Check 1: Module has __file__ starting with user prefix
        if hasattr(mod, '__file__') and mod.__file__:
            if mod.__file__.startswith(user_prefixes):
                modules_to_remove.append(mod_name)
                continue

        # Check 2: Namespace package with __path__ in user directories
        # Namespace packages have __file__ = None but __path__ set
        if hasattr(mod, '__path__'):
            mod_path = getattr(mod, '__path__', None)
            if mod_path:
                # __path__ can be a list or _NamespacePath object
                path_list = list(mod_path) if hasattr(mod_path, '__iter__') else []
                for p in path_list:
                    if isinstance(p, str) and p.startswith(user_prefixes):
                        modules_to_remove.append(mod_name)
                        break

    for mod in modules_to_remove:
        del sys.modules[mod]

    # Also invalidate import caches to handle renamed/moved modules
    importlib.invalidate_caches()

def rebuild_python_path():
    """
    Clear user script paths from sys.path.

    This removes /mnt/pfr and /mnt/blockly from sys.path to prevent
    stale module imports from user code directories.
    """
    sys.path = [
        p for p in sys.path
        if not p.startswith(('/mnt/pfr', '/mnt/blockly'))
    ]

def cleanup_wildcard_imports_and_path(script_dir, exec_globals):
    """
    Clean up wildcard imports from the global namespace.

    This function is called after user code execution to remove all names that were
    imported via wildcard imports (e.g., from module import *), while preserving
    the original initialization imports.

    Args:
        script_dir: The script's directory to remove from sys.path
        exec_globals: The globals dictionary from the main execution context

    Requires the following to be in exec_globals:
    - _wildcard_cleanup_modules: List of module names to clean up (added by transformer)
    - _globals_before_user_code: Set of global names before user code execution

    This prevents namespace pollution between different code executions while maintaining
    the necessary imports from PyodideManager initialization.
    """

    if script_dir and script_dir in sys.path and script_dir != '/mnt/pfr' and script_dir != '/mnt/blockly':
        sys.path.remove(script_dir)

    if '_wildcard_cleanup_modules' not in exec_globals:
        return

    # Check if we have the snapshot of globals before user code
    if '_globals_before_user_code' in exec_globals:
        globals_before = exec_globals['_globals_before_user_code']

        for module_name in exec_globals['_wildcard_cleanup_modules']:
            try:
                # Import the module so we can access it for cleanup
                module = importlib.import_module(module_name)

                # Get all public names from the module
                imported_names = [name for name in dir(module) if not name.startswith('_')]

                # Delete each imported name from globals, but preserve initialization imports
                for name in imported_names:
                    # Only delete if it was added by user code (not present before)
                    if name in exec_globals and name not in globals_before:
                        del exec_globals[name]
            except Exception:
                print("Error during cleanup of module:", module_name)
                pass

        # Clean up the tracking variable
        del exec_globals['_wildcard_cleanup_modules']
    else:
        # If we don't have the snapshot, just delete the tracking variable
        del exec_globals['_wildcard_cleanup_modules']

    # Clean up our snapshot variable
    if '_globals_before_user_code' in exec_globals:
        del exec_globals['_globals_before_user_code']