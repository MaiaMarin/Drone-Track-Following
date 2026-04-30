"""
Custom import system for user modules in Pyodide environment.
Handles import hook installation, path management, and module transformation
"""

import importlib
import importlib.abc
import importlib.machinery
import os
import sys


class TransformingLoader(importlib.abc.Loader):
    """Loader that transforms user modules using transform_imported_module before execution."""

    def __init__(self, fullname, path):
        self.fullname = fullname
        self.path = path

    def get_filename(self, fullname):
        return self.path

    def exec_module(self, module):
        from codrone_edu.tools.transformer import transform_imported_module

        # Read the source file
        with open(self.path, 'r') as f:
            source = f.read()

        # Transform the code for imported modules (no wrapper, no drone init)
        try:
            transformed_ast, transformed_code = transform_imported_module(source)
            # Execute the transformed code in the module's namespace
            code_obj = compile(transformed_ast, self.path, 'exec')
            exec(code_obj, module.__dict__)
        except Exception as e:
            # If transformation fails, execute original code
            try:
                import js
                js.console.log(f"Warning: Could not transform module {self.fullname}: {str(e)}")
                import traceback
                js.console.log(traceback.format_exc())
            except:
                # js module not available (not in Pyodide), just print
                print(f"Warning: Could not transform module {self.fullname}: {str(e)}")

            exec(compile(source, self.path, 'exec'), module.__dict__)


class TransformingFinder(importlib.abc.MetaPathFinder):
    """MetaPathFinder that intercepts imports from user directories."""

    def find_spec(self, fullname, path, target=None):
        # Only transform modules from user directories
        search_paths = path if path is not None else sys.path

        # Early exit: if no user paths to search, skip entirely
        has_user_path = any(
            entry.startswith(('/mnt/pfr', '/mnt/blockly'))
            for entry in search_paths
        )
        if not has_user_path:
            return None

        for entry in search_paths:
            if entry.startswith(('/mnt/pfr', '/mnt/blockly')):
                # Try to find the module file using full dotted path
                parts = fullname.split('.')

                # For submodule imports (e.g., package.module_1), when path is provided
                # and points to the package directory, only use the last part of the name
                if path is not None and len(parts) > 1:
                    # This is a submodule import from within a package
                    # path already points to package dir, so only use the submodule name
                    module_file = os.path.join(entry, parts[-1]) + '.py'
                else:
                    # Top-level import from sys.path
                    # Convert dotted name to file path (e.g., utils.helpers -> utils/helpers.py)
                    module_file = os.path.join(entry, *parts) + '.py'

                if os.path.exists(module_file):
                    loader = TransformingLoader(fullname, module_file)
                    spec = importlib.machinery.ModuleSpec(fullname, loader, origin=module_file)
                    spec.has_location = True
                    return spec

                # Check if it's a package (directory with __init__.py)
                # Only check when importing from sys.path (not when path is a package subdir)
                if path is None:
                    package_init = os.path.join(entry, *parts, '__init__.py')
                    if os.path.exists(package_init):
                        loader = TransformingLoader(fullname, package_init)
                        spec = importlib.machinery.ModuleSpec(fullname, loader, origin=package_init)
                        spec.has_location = True
                        spec.submodule_search_locations = [os.path.join(entry, *parts)]
                        return spec

        return None




def install_import_hook():
    """Install the import hook at the beginning of sys.meta_path."""
    # Check if already installed
    for finder in sys.meta_path:
        if isinstance(finder, TransformingFinder):
            return  # Already installed

    sys.meta_path.insert(0, TransformingFinder())