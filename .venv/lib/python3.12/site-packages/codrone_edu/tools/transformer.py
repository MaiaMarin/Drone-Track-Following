import ast
import astor

# Use frozenset for O(1) membership testing (immutable and hashable)
non_await_drone_functions = frozenset({
    "convert_meter", "convert_millimeter",
    "get_left_joystick_y", "get_left_joystick_x",
    "get_right_joystick_y", "get_right_joystick_x",
    "get_button_data", "l1_pressed", "l2_pressed",
    "r1_pressed", "r2_pressed", "h_pressed",
    "power_pressed", "up_arrow_pressed", "left_arrow_pressed",
    "right_arrow_pressed", "down_arrow_pressed", "s_pressed", "p_pressed",
    "set_roll", "set_pitch", "set_yaw", "set_throttle",
    "print_move_values", "percent_error", "get_move_values",
    "predict_colors", "load_classifier", "print_num_data", "append_color_data", "new_color_data",
    "detect_colors", "load_color_data",
    "reset_classifier", "reset_previous_land"
})

# Common drone methods that should be awaited when called on any variable
# This allows imported modules to use drone instances with any parameter name (e.g., drone_obj)
# Generated from all methods with _emscripten suffix in drone.py
# Use frozenset for O(1) membership testing (immutable and hashable)
awaitable_drone_methods = frozenset({
    # Connection methods (will be replaced with dummy_function) - 11
    "pair", "open", "connect", "disconnect", "close", "reopen", "open_success",
    "isOpen", "isConnected", "receiving", "transfer",
    # Flight control - 5
    "takeoff", "land", "emergency_stop", "hover", "reset_move",
    # Movement - basic - 14
    "go", "move", "turn", "flip", "set_waypoint", "sendLanding", "sendControlWhile",
    "move_distance", "move_forward", "move_backward", "move_left", "move_right",
    "send_absolute_position", "goto_waypoint",
    # Movement - turns - 4
    "turn_degree", "turn_direction", "turn_left", "turn_right",
    # Movement - patterns - 7
    "square", "triangle", "triangle_turn", "spiral", "circle", "circle_turn", "sway",
    # Movement - sensors - 3
    "avoid_wall", "keep_distance", "detect_wall",
    # Buzzer and LED - 15
    "drone_buzzer", "controller_buzzer", "drone_buzzer_sequence", "controller_buzzer_sequence",
    "start_drone_buzzer", "stop_drone_buzzer", "start_controller_buzzer", "stop_controller_buzzer",
    "ping", "set_drone_LED", "set_drone_LED_mode", "drone_LED_off",
    "set_controller_LED", "set_controller_LED_mode", "controller_LED_off",
    # Sensor data getters - 55
    "get_battery", "get_sensor_data", "get_altitude_data", "get_error_data",
    "get_ack_data", "get_pressure", "get_elevation", "get_temperature",
    "get_drone_temperature", "get_range_data", "get_front_range", "get_bottom_range",
    "get_color_data", "get_position_data", "get_pos_x", "get_pos_y", "get_pos_z",
    "get_height", "get_flow_data", "get_flow_velocity_x", "get_flow_velocity_y",
    "get_state_data", "get_system_state", "get_flight_state", "get_movement_state",
    "get_control_speed", "speed_change", "get_motion_data", "get_raw_motion_data",
    "get_accel_x", "get_accel_y", "get_accel_z",
    "get_x_gyro", "get_y_gyro", "get_z_gyro",
    "get_angle_x", "get_angle_y", "get_angle_z",
    "get_joystick_data", "get_trim_data", "get_count", "get_flight_time",
    "get_takeoff_count", "get_landing_count", "get_accident_count",
    "get_cpu_id_data", "get_information_data", "get_address_data", "get_lostconnection_data",
    "get_colors", "get_front_color", "get_back_color", "get_trim",
    # Actions - 6
    "set_initial_pressure", "height_from_pressure", "reset_trim", "reset_gyro",
    "set_trim", "set_motor_speed",
    # Controller display - 14
    "controller_create_canvas", "controller_draw_canvas", "controller_draw_line",
    "controller_draw_rectangle", "controller_draw_square", "controller_draw_point",
    "controller_clear_screen", "controller_draw_polygon", "controller_draw_ellipse",
    "controller_draw_arc", "controller_draw_chord", "controller_draw_string",
    "controller_draw_string_align", "controller_draw_image"
})

# Known library modules that shouldn't be awaited
# Use frozenset for O(1) membership testing and to prevent accidental modification
LIBRARY_MODULES = frozenset({
    'asyncio', 'time', 'sys', 'os', 'math', 'random', 'collections',
    'itertools', 'functools', 'json', 're', 'datetime', 'typing',
    'numpy', 'np', 'scipy', 'pandas', 'matplotlib', 'codrone_edu',
    'colorama', 'PIL', 'cv2', 'sklearn', 'tensorflow', 'torch'
})

class CodeAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.custom_functions = set()
        self.function_aliases = {}  # Maps alias -> original function name
        self.drone_instance_name = None

    def visit_Assign(self, node):
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name):
            if node.value.func.id == 'Drone':
                if node.targets[0].id != 'drone':
                    raise ValueError(f"Drone instance must be named 'drone', found '{node.targets[0].id}' instead.")
                self.drone_instance_name = node.targets[0].id

    def visit_FunctionDef(self, node):
        self.custom_functions.add(node.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        """Track function imports and their aliases"""
        for alias in node.names:
            # Store the mapping: local_name -> original_name
            local_name = alias.asname if alias.asname else alias.name
            self.function_aliases[local_name] = alias.name
        self.generic_visit(node)

    def visit_Import(self, node):
        """Track module imports and their aliases"""
        for alias in node.names:
            # For 'import module as m', we track the alias
            local_name = alias.asname if alias.asname else alias.name
            self.function_aliases[local_name] = alias.name
        self.generic_visit(node)

class AsyncTransformer(ast.NodeTransformer):
    def __init__(self, custom_functions, drone_instance_name, function_aliases=None):
        self.custom_functions = custom_functions
        self.drone_instance_name = drone_instance_name
        self.function_aliases = function_aliases if function_aliases else {}
        self.time_alias = None  # Track what 'time' is imported as
        self.custom_modules = set()  # Track modules from custom imports

    def copy_location(self, new_node, old_node):
        ast.copy_location(new_node, old_node)
        if hasattr(old_node, 'end_lineno'):
            new_node.end_lineno = old_node.end_lineno
        if hasattr(old_node, 'end_col_offset'):
            new_node.end_col_offset = old_node.end_col_offset
        return new_node
    
    def visit_Assign(self, node):
        # Detect assignments like drone = Drone() and comment them out
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name):
            if node.value.func.id == 'Drone' and isinstance(node.targets[0], ast.Name):
                comment_text = f'{node.targets[0].id} = {node.value.func.id}()'
                comment_node = ast.Expr(value=ast.Constant(value=f'# {comment_text}'))
                return self.copy_location(comment_node, node)
        return self.generic_visit(node)

    def visit_Import(self, node):
        """Track time module imports to capture aliasing"""
        for alias in node.names:
            if alias.name == 'time':
                # Store what 'time' is imported as (could be 'time' or an alias like 't')
                self.time_alias = alias.asname if alias.asname else 'time'
        return self.generic_visit(node)

    def visit_ImportFrom(self, node):
        """Track ImportFrom statements"""
        if node.module == 'time':
            # If importing from time module, track it
            self.time_alias = 'time'  # Keep default since it's 'from time import sleep'
        return self.generic_visit(node)

    def is_custom_import(self, node):
        """
        Determine if an import is from custom user code.
        Custom imports are from /mnt/pfr, /mnt/blockly, or relative imports.
        These should be moved inside _wrapper to avoid polluting globals.
        """
        if isinstance(node, ast.ImportFrom):
            # Relative imports (from . import foo, from .. import bar)
            if node.level > 0:
                return True
            # Check if module path indicates custom code
            if node.module:
                # Check for paths that would be handled by module_loader.py
                # Note: We check the module name, not the full path
                parts = node.module.split('.')
                # If it doesn't look like a standard library or known package, treat as custom
                # This is a heuristic - custom modules typically don't have common library names
                if parts[0] not in LIBRARY_MODULES:
                    return True
        elif isinstance(node, ast.Import):
            # Direct imports: import my_module
            # Check each imported name
            for alias in node.names:
                parts = alias.name.split('.')
                if parts[0] not in LIBRARY_MODULES:
                    return True
        return False

    def visit_Module(self, node):
        # Separate imports into library vs custom, and other statements
        library_imports = []
        custom_imports = []
        wildcard_module_imports = []  # Module-level wildcard imports
        wildcard_modules_to_cleanup = []  # Track module names for cleanup
        other_statements = []

        for n in node.body:
            if isinstance(n, (ast.Import, ast.ImportFrom)):
                # Track time alias before categorizing imports
                if isinstance(n, ast.Import):
                    for alias in n.names:
                        if alias.name == 'time':
                            self.time_alias = alias.asname if alias.asname else 'time'
                elif isinstance(n, ast.ImportFrom):
                    if n.module == 'time':
                        self.time_alias = 'time'

                if self.is_custom_import(n):
                    # Check if this is a wildcard import (from module import *)
                    is_wildcard = isinstance(n, ast.ImportFrom) and any(alias.name == '*' for alias in n.names)

                    if is_wildcard:
                        # Wildcard imports must stay at module level (Python syntax requirement)
                        # Keep the original import statement
                        wildcard_module_imports.append(n)
                        # Track the module name for cleanup
                        wildcard_modules_to_cleanup.append(n.module)
                    else:
                        custom_imports.append(n)

                    # Track custom module names
                    if isinstance(n, ast.ImportFrom):
                        # For "from module import x", track imported names
                        for alias in n.names:
                            local_name = alias.asname if alias.asname else alias.name
                            if local_name != '*':  # Skip import *
                                self.custom_modules.add(local_name)
                            else:
                                # For wildcard imports, track the module name
                                self.custom_modules.add(n.module)
                    elif isinstance(n, ast.Import):
                        # For "import module", track module name
                        for alias in n.names:
                            local_name = alias.asname if alias.asname else alias.name
                            self.custom_modules.add(local_name)
                else:
                    library_imports.append(n)
            else:
                other_statements.append(n)

        # Remove extra newlines in other_statements
        other_statements = [stmt for stmt in other_statements if not (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and stmt.value.value == '')]

        # Insert drone.reset_classifier() at the beginning
        reset_classifier_call = ast.Expr(
            value=ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id=self.drone_instance_name, ctx=ast.Load()),
                    attr="reset_classifier",
                    ctx=ast.Load(),
                ),
                args=[],
                keywords=[],
            )
        )

        reset_previous_land_call = ast.Expr(
            value=ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id=self.drone_instance_name, ctx=ast.Load()),
                    attr="reset_previous_land",
                    ctx=ast.Load(),
                ),
                args=[],
                keywords=[],
            )
        )

        # Build wrapper body: custom imports (non-wildcard) + reset calls + other statements
        wrapper_body = custom_imports + [reset_classifier_call, reset_previous_land_call] + other_statements

        # Wrap the rest of the code in _wrapper function
        wrapper_function = ast.AsyncFunctionDef(
            name='_wrapper',
            args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
            body=wrapper_body,
            decorator_list=[]
        )

        # Store wildcard module names in a global variable for cleanup
        # This will be accessed by PyodideManager to clean up after execution
        wildcard_cleanup_list = []
        if wildcard_modules_to_cleanup:
            # Create a global variable to store module names for cleanup
            cleanup_list_code = f"_wildcard_cleanup_modules = {wildcard_modules_to_cleanup}"
            wildcard_cleanup_list = ast.parse(cleanup_list_code).body

        # Module structure: library imports + wildcard imports + cleanup list + wrapper
        # Wildcard imports must remain at module level due to Python syntax restrictions
        # The cleanup list is stored in a global variable for post-execution cleanup
        node.body = library_imports + wildcard_module_imports + wildcard_cleanup_list + [wrapper_function]
        self.generic_visit(wrapper_function)  # Visit the wrapped code inside _wrapper
        return node
    
    def visit_FunctionDef(self, node):
        new_node = ast.AsyncFunctionDef(
            name=node.name,
            args=node.args,
            body=node.body,
            decorator_list=node.decorator_list,
            returns=node.returns
        )
        new_node = self.copy_location(new_node, node)
        self.generic_visit(new_node)
        return new_node
    
    def visit_Call(self, node):
        # First, visit all arguments and keywords to ensure nested calls are transformed
        node = self.generic_visit(node)

        if isinstance(node.func, ast.Attribute):
            # Handle attribute calls (e.g., drone.takeoff(), helper.do_something(), np.array())
            if isinstance(node.func.value, ast.Name):
                module_name = node.func.value.id
                method_name = node.func.attr

                # Check if this is a drone method call (by method name, not variable name)
                # This allows imported modules to use drone instances with any parameter name
                if method_name in awaitable_drone_methods:
                    if method_name in ['pair', 'open', 'connect', 'disconnect', 'close']:
                        # Replace with dummy_function()
                        new_node = ast.Call(
                            func=ast.Attribute(value=node.func.value, attr='dummy_function', ctx=ast.Load()),
                            args=[], keywords=[]
                        )
                        return self.copy_location(new_node, node)
                    elif method_name not in non_await_drone_functions:
                        new_node = ast.Await(value=node)
                        return self.copy_location(new_node, node)
                elif module_name == self.drone_instance_name:
                    # Fallback: check if variable is named 'drone' (for any unlisted methods)
                    if node.func.attr in ['pair', 'open', 'connect', 'disconnect', 'close']:
                        # Replace with dummy_function()
                        new_node = ast.Call(
                            func=ast.Attribute(value=ast.Name(id=self.drone_instance_name, ctx=ast.Load()), attr='dummy_function', ctx=ast.Load()),
                            args=[], keywords=[]
                        )
                        return self.copy_location(new_node, node)
                    elif node.func.attr not in non_await_drone_functions:
                        new_node = ast.Await(value=node)
                        return self.copy_location(new_node, node)
                elif module_name in self.custom_modules:
                    # Calls on custom imported modules (e.g., helper.do_something())
                    new_node = ast.Await(value=node)
                    return self.copy_location(new_node, node)
                # Skip library module calls (e.g., np.array(), asyncio.sleep())
                # Note: asyncio.sleep() will already have await added by SleepTransformer
        elif isinstance(node.func, ast.Name):
            # Direct function calls (e.g., my_function(), some_utility_function())
            func_name = node.func.id

            # Check if it's a Python built-in function (don't await built-ins)
            # EXCEPT for input() which is overridden with an async version in Pyodide
            import builtins
            if hasattr(builtins, func_name) and func_name != 'input':
                builtin_obj = getattr(builtins, func_name)
                # Check if it's callable (a function/class) and not async
                # Exclude async built-ins like aiter, anext
                if callable(builtin_obj) and func_name not in {'aiter', 'anext'}:
                    return node

            # Look up the original function name if this is an alias
            original_name = self.function_aliases.get(func_name, func_name)

            # Await if it's NOT a known library function
            # This handles:
            # - Custom functions defined in the file
            # - Functions imported from custom modules
            # - Functions from 'import *' (which we can't track)
            # - input() calls (overridden with async version in Pyodide)
            if (func_name in self.custom_functions or
                original_name in self.custom_functions or
                func_name in self.custom_modules or
                func_name == 'input' or
                func_name not in LIBRARY_MODULES):  # Await unknown functions (could be from import *)
                new_node = ast.Await(value=node)
                return self.copy_location(new_node, node)
        return node

class SleepTransformer(ast.NodeTransformer):
    def __init__(self, time_alias=None):
        self.time_alias = time_alias if time_alias else 'time'
        self.has_time_import = False
        self.asyncio_imported = False

    def copy_location(self, new_node, old_node):
        ast.copy_location(new_node, old_node)
        if hasattr(old_node, 'end_lineno'):
            new_node.end_lineno = old_node.end_lineno
        if hasattr(old_node, 'end_col_offset'):
            new_node.end_col_offset = old_node.end_col_offset
        return new_node

    def visit_Module(self, node):
        # First pass: check if time is imported and add asyncio if needed
        has_time = False
        has_asyncio = False

        for n in node.body:
            if isinstance(n, ast.Import):
                for alias in n.names:
                    if alias.name == 'time':
                        has_time = True
                    elif alias.name == 'asyncio':
                        has_asyncio = True
            elif isinstance(n, ast.ImportFrom):
                if n.module == 'time':
                    has_time = True
                elif n.module == 'asyncio':
                    has_asyncio = True

        # If time is imported but asyncio is not, add asyncio import
        if has_time and not has_asyncio:
            asyncio_import = ast.Import(names=[ast.alias(name='asyncio', asname=None)])
            # Insert asyncio import after the time import
            new_body = []
            inserted = False
            for n in node.body:
                new_body.append(n)
                if not inserted and isinstance(n, (ast.Import, ast.ImportFrom)):
                    # Check if this is the time import
                    is_time_import = False
                    if isinstance(n, ast.Import):
                        is_time_import = any(alias.name == 'time' for alias in n.names)
                    elif isinstance(n, ast.ImportFrom):
                        is_time_import = n.module == 'time'

                    if is_time_import:
                        new_body.append(asyncio_import)
                        inserted = True

            node.body = new_body

        # Now continue with normal transformation
        self.generic_visit(node)
        return node

    def visit_ImportFrom(self, node):
        # Keep 'from time import ...' as-is, don't transform to asyncio
        # The user might be importing time.time() or other functions
        return node

    def visit_Call(self, node):
        # Convert time.sleep calls to asyncio.sleep (with await)
        # Keep time.time() and other time functions as-is
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            # Check if this matches the time alias (could be 'time', 't', etc.)
            if node.func.value.id == self.time_alias and node.func.attr == 'sleep':
                # Change to asyncio.sleep
                new_call = ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id='asyncio', ctx=ast.Load()),
                        attr='sleep',
                        ctx=ast.Load()
                    ),
                    args=node.args,
                    keywords=node.keywords
                )
                new_node = ast.Await(value=new_call)
                return self.copy_location(new_node, node)
        return self.generic_visit(node)

class CheckInterruptAndGlobalTransformer(ast.NodeTransformer):
    def __init__(self, convert_global_to_nonlocal=True):
        self.convert_global_to_nonlocal = convert_global_to_nonlocal

    def copy_location(self, new_node, old_node):
        ast.copy_location(new_node, old_node)
        if hasattr(old_node, 'end_lineno'):
            new_node.end_lineno = old_node.end_lineno
        if hasattr(old_node, 'end_col_offset'):
            new_node.end_col_offset = old_node.end_col_offset
        return new_node

    def generic_visit(self, node):
        if isinstance(node, ast.stmt) and not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.If, ast.For, ast.While, ast.Try)):
            check_interrupt_call = ast.Expr(value=ast.Call(
                func=ast.Name(id='checkInterrupt', ctx=ast.Load()),
                args=[], keywords=[]
            ))
            check_interrupt_call = self.copy_location(check_interrupt_call, node)
            return [node, check_interrupt_call]
        return super().generic_visit(node)

    def visit_Global(self, node):
        # Only convert global to nonlocal if we're in a wrapper function
        if self.convert_global_to_nonlocal:
            nonlocal_node = ast.Nonlocal(names=node.names)
            return ast.copy_location(nonlocal_node, node)
        return node

def transform_code(code):
    # Parse the code into an AST
    tree = ast.parse(code)

    analyzer = CodeAnalyzer()
    analyzer.visit(tree)

    # Transform the AST - pass function_aliases to AsyncTransformer
    async_transformer = AsyncTransformer(
        analyzer.custom_functions,
        "drone",
        analyzer.function_aliases
    )
    tree = async_transformer.visit(tree)

    # Create SleepTransformer with the time_alias from AsyncTransformer
    sleep_transformer = SleepTransformer(async_transformer.time_alias)
    # Create combined transformer with global-to-nonlocal conversion enabled (for wrapper function)
    combined_transformer = CheckInterruptAndGlobalTransformer(convert_global_to_nonlocal=True)

    # SleepTransformer runs on entire tree to transform time imports at module level
    tree = sleep_transformer.visit(tree)

    # CheckInterrupt and Global transformers only run on _wrapper function body
    for node in tree.body:
        if isinstance(node, ast.AsyncFunctionDef) and node.name == '_wrapper':
            combined_transformer.visit(node)
            break

    # Ensure the tree is correctly fixed
    ast.fix_missing_locations(tree)

    # Convert the modified AST back to code
    # modified_code = ast.unparse(tree)
    modified_code = astor.to_source(tree)
    return tree, modified_code


def transform_imported_module(code):
    """
    Transform imported modules without wrapping in _wrapper function.
    This is used for user-created modules that are imported, not executed directly.
    Injects checkInterrupt import at the beginning to enable interruptibility.
    """
    # Parse the code into an AST
    tree = ast.parse(code)

    analyzer = CodeAnalyzer()
    analyzer.visit(tree)

    # Create a modified AsyncTransformer that doesn't wrap in _wrapper
    class ModuleAsyncTransformer(ast.NodeTransformer):
        def __init__(self, custom_functions, drone_instance_name, function_aliases=None):
            self.custom_functions = custom_functions
            self.drone_instance_name = drone_instance_name
            self.function_aliases = function_aliases if function_aliases else {}
            self.time_alias = None  # Track what 'time' is imported as
            self.custom_modules = set()  # Track modules from custom imports

        def copy_location(self, new_node, old_node):
            ast.copy_location(new_node, old_node)
            if hasattr(old_node, 'end_lineno'):
                new_node.end_lineno = old_node.end_lineno
            if hasattr(old_node, 'end_col_offset'):
                new_node.end_col_offset = old_node.end_col_offset
            return new_node

        def visit_Import(self, node):
            """Track time module imports and custom module imports"""
            for alias in node.names:
                if alias.name == 'time':
                    self.time_alias = alias.asname if alias.asname else 'time'
                else:
                    # Track custom module imports (not library modules)
                    module_name = alias.name.split('.')[0]  # Get root module name
                    if module_name not in LIBRARY_MODULES:
                        local_name = alias.asname if alias.asname else alias.name
                        self.custom_modules.add(local_name)
            return self.generic_visit(node)

        def visit_ImportFrom(self, node):
            """Track ImportFrom statements for both time and custom modules"""
            if node.module == 'time':
                self.time_alias = 'time'
            elif node.module:
                # Track custom module imports
                module_name = node.module.split('.')[0]  # Get root module name
                if module_name not in LIBRARY_MODULES:
                    # For "from module import x", track imported names
                    for alias in node.names:
                        local_name = alias.asname if alias.asname else alias.name
                        if local_name != '*':  # Skip import *
                            self.custom_modules.add(local_name)
            return self.generic_visit(node)

        # Don't wrap module - just transform functions
        def visit_FunctionDef(self, node):
            new_node = ast.AsyncFunctionDef(
                name=node.name,
                args=node.args,
                body=node.body,
                decorator_list=node.decorator_list,
                returns=node.returns
            )
            new_node = self.copy_location(new_node, node)
            self.generic_visit(new_node)
            return new_node

        def visit_Call(self, node):
            # First, visit all arguments and keywords to ensure nested calls are transformed
            node = self.generic_visit(node)

            if isinstance(node.func, ast.Attribute):
                # Handle attribute calls (e.g., drone.takeoff(), helper.do_something(), np.array())
                if isinstance(node.func.value, ast.Name):
                    module_name = node.func.value.id
                    method_name = node.func.attr

                    # Check if this is a drone method call (by method name, not variable name)
                    # This allows imported modules to use drone instances with any parameter name
                    if method_name in awaitable_drone_methods:
                        if method_name in ['pair', 'open', 'connect', 'disconnect', 'close']:
                            # Replace with dummy_function()
                            new_node = ast.Call(
                                func=ast.Attribute(value=node.func.value, attr='dummy_function', ctx=ast.Load()),
                                args=[], keywords=[]
                            )
                            return self.copy_location(new_node, node)
                        elif method_name not in non_await_drone_functions:
                            new_node = ast.Await(value=node)
                            return self.copy_location(new_node, node)
                    elif module_name == self.drone_instance_name:
                        # Fallback: check if variable is named 'drone' (for any unlisted methods)
                        if node.func.attr in ['pair', 'open', 'connect', 'disconnect', 'close']:
                            # Replace with dummy_function()
                            new_node = ast.Call(
                                func=ast.Attribute(value=ast.Name(id=self.drone_instance_name, ctx=ast.Load()), attr='dummy_function', ctx=ast.Load()),
                                args=[], keywords=[]
                            )
                            return self.copy_location(new_node, node)
                        elif node.func.attr not in non_await_drone_functions:
                            new_node = ast.Await(value=node)
                            return self.copy_location(new_node, node)
                    elif module_name in self.custom_modules:
                        # Calls on custom imported modules (e.g., helper.do_something())
                        new_node = ast.Await(value=node)
                        return self.copy_location(new_node, node)
                    # Skip library module calls (e.g., np.array(), asyncio.sleep())
            elif isinstance(node.func, ast.Name):
                # Direct function calls (e.g., my_function(), some_utility_function())
                func_name = node.func.id

                # Check if it's a Python built-in function (don't await built-ins)
                # EXCEPT for input() which is overridden with an async version in Pyodide
                import builtins
                if hasattr(builtins, func_name) and func_name != 'input':
                    builtin_obj = getattr(builtins, func_name)
                    # Check if it's callable (a function/class) and not async
                    # Exclude async built-ins like aiter, anext
                    if callable(builtin_obj) and func_name not in {'aiter', 'anext'}:
                        return node

                # Look up the original function name if this is an alias
                original_name = self.function_aliases.get(func_name, func_name)

                # Await if it's NOT a known library function
                # This handles:
                # - Custom functions defined in the file
                # - Functions imported from custom modules
                # - Functions from 'import *' (which we can't track)
                # - input() calls (overridden with async version in Pyodide)
                if (func_name in self.custom_functions or
                    original_name in self.custom_functions or
                    func_name in self.custom_modules or
                    func_name == 'input' or
                    func_name not in LIBRARY_MODULES):  # Await unknown functions (could be from import *)
                    new_node = ast.Await(value=node)
                    return self.copy_location(new_node, node)
            return node

    # Transform the AST (without module wrapper)
    module_transformer = ModuleAsyncTransformer(
        analyzer.custom_functions,
        "drone",
        analyzer.function_aliases
    )
    tree = module_transformer.visit(tree)

    # Create SleepTransformer with the time_alias from ModuleAsyncTransformer
    sleep_transformer = SleepTransformer(module_transformer.time_alias)
    # Create combined transformer WITHOUT global-to-nonlocal conversion (module-level code)
    combined_transformer = CheckInterruptAndGlobalTransformer(convert_global_to_nonlocal=False)

    tree = sleep_transformer.visit(tree)
    tree = combined_transformer.visit(tree)

    # Inject checkInterrupt import at the beginning of the module
    # This is necessary because CheckInterruptTransformer adds checkInterrupt() calls
    check_interrupt_import = ast.ImportFrom(
        module='codrone_edu.tools.interrupter',
        names=[ast.alias(name='*', asname=None)],
        level=0
    )

    # Insert the import at the beginning of the module
    tree.body.insert(0, check_interrupt_import)

    # Ensure the tree is correctly fixed
    ast.fix_missing_locations(tree)

    # Convert the modified AST back to code
    modified_code = astor.to_source(tree)
    return tree, modified_code


# test="""
# dataset = "color_data"
# colors = ["green", "red", "blue", "yellow"]
# for color in colors:
#     data = []
#     samples = 500
#     for i in range(1):
#         print("Sample: ", i+1)
#         next = input("Press enter to calibrate " + color)
#         print("0% ", end="")
#         for j in range(samples):
#             color_data = drone.get_color_data()[0:9]
#             data.append(color_data)
#             time.sleep(0.005)
#             if j % 10 == 0:
#                 print("-", end="")
#         print(" 100%")
#     drone.new_color_data(color, data, dataset)
# print("Done calibrating.")
# """

# transformed_ast, transformed_code = transform_code(test)
# print(transformed_code)

# code_object = compile(transformed_ast, filename="<ast>", mode="exec")
# print(code_object)
# print(transformed_ast)