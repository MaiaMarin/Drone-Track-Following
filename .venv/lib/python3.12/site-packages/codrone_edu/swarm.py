import importlib.metadata
from serial.tools import list_ports
from codrone_edu.drone import *
import time
import asyncio


class Swarm:
    """
    This will instantiate an object, creating an instance of the Swarm class, which is used to connect to multiple
    drones and allow the connected drones to perform commands at the same time or different times.

    :param enable_color: Boolean value, enables LED color assignment for each drone. True by default
    :param enable_print: Boolean value, enables console output related to drone LED assignment and index. True by default
    :param enable_pause: Boolean value, enables pause on program by prompting user to press Enter to continue. True by default

    :return: Swarm object
    """
    def __init__(self, enable_color=True, enable_print=True, enable_pause=True):
        self._enable_color = enable_color
        self._enable_print = enable_print
        self._enable_pause = enable_pause
        self._drone_objects = []
        self._num_drones = 0
        self._print_lock = asyncio.Lock()
        self._portnames = []
        self._led_colors = ['red', 'blue', 'orange', 'yellow', 'green', 'light blue', 'purple', 'pink', 'white', 'black']
        self._rgb_colors = [[255, 0, 0, 255],[25, 116, 210, 255],[255, 140, 0, 255],[255, 255, 0, 255],[0, 255, 0, 255],[172, 229, 238, 255],
                        [255, 0, 255, 255],[255, 255, 255, 255],[0, 0, 0, 255]]

    def __getitem__(self, subscript):
        if isinstance(subscript, slice):
            return SwarmFragment(self._drone_objects[subscript.start:subscript.stop:subscript.step])
        else:
            return SwarmFragment(self._drone_objects[subscript])

    def __getattr__(self, item, *args, **kwargs):

        def call_method(*args, **kwargs):
            return self.all_drones(item, *args, **kwargs)

        return call_method

    ## Swarm printing start ##
    def print(self, format="raw"):
        swarm_str = 'Swarm\n'

        if format == "raw":
            swarm_str += f"{self._drone_objects}"
            swarm_str += "\n"

        elif format == "readable":
            for i in range(self._num_drones):
                swarm_str += f"\tDrone {i} at port {self._portnames[i]}"
                if self._enable_color:
                    swarm_str += f": {self._led_colors[i]}"
                swarm_str += "\n"
        else:
            print(Fore.YELLOW + "Warning: Invalid format. Valid formats are 'raw' and 'readable'." + Style.RESET_ALL)
            return

        print(swarm_str[:-1])
    ## Swarm printing end ##

    ## Swarm Connect Start ##
    async def _connect(self):
        x = list(list_ports.comports(include_links=True))

        async def check_port(element):
            async with self._print_lock:
                if element.vid == 1155 or element.vid == 6790:
                    portname = element.device
                    self._portnames.append(str(portname))

        await asyncio.gather(*(check_port(element) for element in x))

        self._num_drones = len(self._portnames)

        async def create_drone():
            self._drone_objects.append(Drone(swarm=True))

        await asyncio.gather(*(create_drone() for _ in range(self._num_drones)))

        colorama.init()
        library_name = "codrone-edu"
        library = importlib.metadata.distribution(library_name).version
        print(Fore.GREEN + f"Running {library_name} library version {library}" + Style.RESET_ALL)

        await asyncio.gather(*(self._initialize_drone(i) for i in range(self._num_drones)))

        await asyncio.gather(*(self._connect_drone(i, self._portnames[i]) for i in range(self._num_drones)))

    def connect(self):
        """
        This function connects all the CoDroneEDU controllers with the computer

        :return: None
        """
        asyncio.run(self._connect())
        if self._enable_print and self._enable_color:
            print()
            for i in range(self._num_drones):
                print(Fore.GREEN + f"Drone {i} at port {self._portnames[i]}: {self._led_colors[i]}" + Style.RESET_ALL)
        if self._enable_pause:
            input("Press Enter to start swarm...")


    async def _connect_drone(self, index, portname):
        await self._drone_objects[index].pair(portname)
        if self._enable_color:
            await self._drone_objects[index].set_drone_LED(*self._rgb_colors[index])
            await self._drone_objects[index].set_controller_LED(*self._rgb_colors[index])

    async def _initialize_drone(self, index):
        await self._drone_objects[index].initialize_data()

    ## Swarm Connect End ##

    async def __call_method(self, index, command):
        drone = self._drone_objects[index]
        method_name = command[0]
        args = command[1]
        kwargs = command[2]
        method = getattr(drone, method_name, None)
        if callable(method):
            try:
                return await method(*args, **kwargs)
            except TypeError:
                return method(*args, **kwargs)  # this is the case for non-async functions like set_yaw
        else:
            async with self._print_lock:
                print("Method ", method_name, " not found")
                return

    ## Basic Swarm Start ##
    async def _all_drones(self, method_name, *args, **kwargs):
        command = [method_name, args, kwargs]

        results = await asyncio.gather(*(self.__call_method(i, command) for i in range(self._num_drones)))

        return results

    def all_drones(self, method_name, *args, **kwargs):
        return asyncio.run(self._all_drones(method_name, *args, **kwargs))

    async def _one_drone(self, index, method_name, *args, **kwargs):
        command = [method_name, args, kwargs]

        # without subscript, returns a list of actual value, which we don't need since it's only one drone
        # with subscript, just returns the actual value from drone function
        return (await asyncio.gather(self.__call_method(index, command)))[0]

    def one_drone(self, index, method_name, *args, **kwargs):
        return asyncio.run(self._one_drone(index, method_name, *args, **kwargs))

    # wrapper function for one_drone()
    def run_drone(self, index, method_name, *args, **kwargs):
        """
        This function runs a Drone function for only one drone in the swarm.

        :param index: index of desired Drone that will execute Drone function
        :param method_name: string value, name of the Drone function that will be executed
        :param *args: arguments for Drone function, optional depending on the Drone function
        :param **kwargs: keyword arguments for Drone function, optional

        :return: (Depends on what Drone function is called)
        """
        return self.one_drone(index, method_name, *args, **kwargs)

    ## Basic Swarm End ##


    ## Run Sync Start ##
    # Slightly different functionality than _one_drone, this runs multiple commands for one drone
    async def _run_one_drone(self, index, commands, max_num_commands):
        return_values = []
        for com in commands:
            return_value = await self.__call_method(index, com)
            return_values.append(return_value)

        reset_command = ['reset_move_values', (), {}]
        await self.__call_method(index, reset_command)

        # number of None values to append to make list for each drone the same length, necessary to transpose 2D list in _run()
        n_blank = abs(len(commands) - max_num_commands)

        for _ in range(n_blank):
            return_values.append(None)

        return return_values

    async def _run(self, sync_obj, type="parallel", delay=None, order=None):

        sync_tasks = sync_obj.get_sync()
        num_synced = len(sync_tasks)  # number of drones involved in sync
        max_steps = sync_obj.get_max_num_steps()  # gets max number of tasks to perform out of all the synced drones

        if num_synced > self._num_drones:
            await self._all_drones('land')
            await self._disconnect()
            raise InvalidSyncError('Number of drones required for sync is higher than number of drones connected!')


        if order is not None:
            if len(order) > max_steps:
                await self._all_drones('land')
                await self._disconnect()
                raise InvalidOrderError(f'len(order) ({len(order)}) is greater than the max number of tasks in sync ({max_steps} task(s))')

        if (type == "sequential") and (num_synced > 1):
            if delay is None:
                delay = 0

            if order is None:
                # if no order is specified, the order will go by increasing drone index
                # order = [
                # [0,1,2,...,n-1], method_name1
                # [0,1,2,...,n-1], method_name2
                # ...,
                # [0,1,2,...,n-1] method_namei
                # ]
                order = [[i for i in range(num_synced)] for _ in range(max_steps)]

            return_values = []

            for i in range(max_steps):

                temp_return_values = []

                for index in order[i]:

                    if i <= len(sync_tasks[index]) - 1:
                        method_name = sync_tasks[index][i][0]
                        args = sync_tasks[index][i][1]
                        kwargs = sync_tasks[index][i][2]

                    # if drone doesn't have any more scheduled tasks, it will just hover
                    else:
                        method_name = 'reset_move_values'
                        args = []
                        kwargs = {}

                    return_value = await self._one_drone(index, method_name, *args, **kwargs)
                    temp_return_values.append(return_value)
                    time.sleep(delay)

                return_values.append(temp_return_values)

            return return_values
        # run if type="parallel", each drone will perform their sequence individually
        else:
            tasks = []

            for index, commands in sync_tasks.items():
                task = asyncio.create_task(self._run_one_drone(index, commands, max_steps))
                tasks.append(task)

            return_values = await asyncio.gather(*tasks)

            transposed_return_values = [list(row) for row in zip(*return_values)]
            return transposed_return_values

    def run(self, sync_obj, type="parallel", delay=None, order=None):
        """
        Runs each of the drone's sequences independently, at the same time, or runs each of the drone's sequences one by one.

        :param sync_obj: Sync value, Sync object that will be executed.
        :param type: string value, type of synchronization ("sequential" and "parallel"). "parallel" by default.
        :param delay: float value, delay between drone commands, if type="sequential".
        :param order: 2D list value, order of execution between drones, if type="sequential".

        :return: 2D list value, drone data from every command ran for each drone.
        """
        return asyncio.run(self._run(sync_obj, type, delay, order))

    ## Run Sync End ##

    def get_drones(self):
        return self._drone_objects

    async def _disconnect_drone(self, index):
        await self._drone_objects[index].disconnect()

    async def _disconnect(self):
        await asyncio.gather(*(self._disconnect_drone(i) for i in range(self._num_drones)))

    def close(self):
        asyncio.run(self._disconnect())

    def disconnect(self):
        """
        This function connects all the CoDroneEDU controllers with the computer

        :return: None
        """
        asyncio.run(self._disconnect())

class SwarmFragment:
    """
    This will instantiate an object, creating an instance of the SwarmFragment class. Created from slicing or indexing a
    Swarm object, in order to perform actions like swarm[1].turn_right() or swarm[:2].move_forward().

    NOTE: This class does not have all of Swarm's functionalities. Utilize Swarm class to have full control of the swarm.

    :param drone_objects: list value, list of Drone objects created from Swarm object

    :return: SwarmFragment object
    """
    def __init__(self, drone_objects):
        self._drone_objects = drone_objects if isinstance(drone_objects, list) else [drone_objects]
        self._num_drones = len(self._drone_objects)

    def __getattr__(self, item, *args, **kwargs):
        def call_method(*args, **kwargs):
            return self.all_drones(item, *args, **kwargs)

        return call_method

    def __getitem__(self, subscript):
        if isinstance(subscript, slice):
            return SwarmFragment(self._drone_objects[subscript.start:subscript.stop:subscript.step])
        else:
            return SwarmFragment(self._drone_objects[subscript])

    async def __call_method(self, index, command):
        drone = self._drone_objects[index]
        method_name = command[0]
        args = command[1]
        kwargs = command[2]
        method = getattr(drone, method_name, None)
        if callable(method):
            try:
                return await method(*args, **kwargs)
            except TypeError:
                return method(*args, **kwargs)  # this is the case for non-async functions like set_yaw
        else:
            async with self._print_lock:
                print("Method ", method_name, " not found")
                return

    async def _all_drones(self, method, *args, **kwargs):
        command = [method, args, kwargs]

        results = await asyncio.gather(*(self.__call_method(i, command) for i in range(self._num_drones)))

        return results

    def all_drones(self, method, *args, **kwargs):
        return asyncio.run(self._all_drones(method, *args, **kwargs))



## Sync Class Start ##
class Sync:
    """
    This instantiates an object, creating an instance of the Sync class, which is used to store Sequence objects from
    each drone in order to synchronize the drones with swarm.run().

    :param Sequence objects: Sequence value(s), Sequence objects to store. If adding multiple, separate by a comma.

    :return: Sync object
    """
    def __init__(self, *args):
        self._sync = {}
        # should look like this:
        # {0: [[method_name1,args1,kwargs1], [method_name2,args2,kwargs2]],
        #  1: [[method_name1,args1,kwargs1]],
        #  2: [[method_name1,args1,kwargs1], [method_name2,args2,kwargs2], [method_name3,args3,kwargs3]]}

        # process variable number of sequences and store into self._sync
        for sequence_obj in args:
            sequence_dict = sequence_obj.get_sequence()
            sequence = list(sequence_dict.items()) # convert dictionary iterable into list of key-val tuples
            index, method_list = sequence[0][0], sequence[0][1]
            for method in method_list:
                if index not in self._sync:
                    self._sync[index] = [method]
                else:
                    self._sync[index].append(method)

    def print(self, format="raw"):
        sync_str = ''

        if format == "raw":
            sync_str += '{'
            for sequence in self._sync.items():
                drone_str = f'{sequence[0]}'
                commands_str = f'{sequence[1]}'
                sync_str += drone_str + ': ' + commands_str + ',\n'

            sync_str = sync_str[:-2] # remove last ',\n'
            sync_str += '}'

        elif format == "readable":
            for sequence in self._sync.items():
                # sequence -> (index, [[method_name1,args1,kwargs1], [method_name2,args2,kwargs2]])
                drone_str = f'Drone {sequence[0]}\n'
                commands_str = ''
                commands = sequence[1]  # list of function calls
                for command in commands:
                    args_str = ', '.join(str(arg) for arg in command[1])
                    kwargs_str = ''
                    kwargs_list = list(command[2].items())

                    # if kwargs exist
                    if len(kwargs_list) > 0:
                        if len(command[1]) > 0: kwargs_str += ', '  # if args and kwargs both exist

                        kwargs_str += ', '.join(kwarg[0] + '=' + str(kwarg[1]) for kwarg in kwargs_list)

                    # concatenating '- function(a_1, a_2,..., a_n, kwa_1=val_1, kwa_2=val_2,..., kwa_n=val_n)'
                    commands_str += f'\t- {command[0]}({args_str + kwargs_str})\n'

                commands_str += '\n'

                sync_str += (drone_str + commands_str)
                sync_str = sync_str[:-2] # remove last '\n' character
        else:
            print(Fore.YELLOW + "Warning: Invalid format. Valid formats are 'raw' and 'readable'." + Style.RESET_ALL)
            return

        print(sync_str)

    def add(self, sequence_obj):
        """
        This function adds a Sequence object in the Sync object.

        :param sequence_obj: Sequence value, Sequence object to be added.

        :return: None
        """
        sequence_dict = sequence_obj.get_sequence()
        sequence = list(sequence_dict.items())  # convert dictionary iterable into list of key-val tuples
        index, method_list = sequence[0][0], sequence[0][1]
        for method in method_list:
            if index not in self._sync:
                self._sync[index] = [method]
            else:
                self._sync[index].append(method)

    def get_sync(self):
        return self._sync

    def get_size(self):
        return len(self._sync)

    def get_max_num_steps(self):
        max_steps = 0
        if len(self._sync) > 0:
            for k,v in self._sync.items():
                curr_steps = len(v)
                if curr_steps > max_steps:
                    max_steps = curr_steps
            return max_steps
        else:
            return 0

## Sync Class End ##

## Sequence Class Start ##
class Sequence:
    """
    This instantiates an object, creating an instance of the Sequence class, which is used to schedule a sequence of
    drone commands for a given drone.

    :param index: integer value, the index of the drone.

    :returns: A sequence object that can schedule drone commands for a given drone.
    """
    def __init__(self, index):
        self.index = index
        self._sequence = {index: []}
        # should look like this:
        # {0: [[method_name1,args1,kwargs1], [method_name2,args2,kwargs2]]}

    def print(self, format='raw'):
        seq_str = ''

        if format=='raw':
            seq_str += '{'
            drone_str = f'{self.index}'
            commands_str = f'{self._sequence[self.index]}'
            seq_str += drone_str + ': ' + commands_str + ',\n'

            seq_str = seq_str[:-2]  # remove last ',\n'
            seq_str += '}'

        elif format=='readable':
            # sequence -> (index, [[method_name1,args1,kwargs1], [method_name2,args2,kwargs2]])
            drone_str = f'Drone {self.index}\n'
            commands_str = ''
            commands = self._sequence[self.index] # list of function calls
            for command in commands:
                args_str = ', '.join(str(arg) for arg in command[1])
                kwargs_str = ''
                kwargs_list = list(command[2].items())

                # if kwargs exist
                if len(kwargs_list) > 0:
                    if len(command[1]) > 0: kwargs_str += ', ' # if args and kwargs both exist

                    kwargs_str += ', '.join(kwarg[0] + '=' + str(kwarg[1]) for kwarg in kwargs_list)

                # concatenating '- function(a_1, a_2,..., a_n, kwa_1=val_1, kwa_2=val_2,..., kwa_n=val_n)'
                commands_str += f'\t- {command[0]}({args_str + kwargs_str})\n'

            seq_str += (drone_str + commands_str)

            # remove last '\n' character
            seq_str = seq_str[:-1]

        else:
            print(Fore.YELLOW + "Warning: Invalid format. Valid formats are 'raw' and 'readable'." + Style.RESET_ALL)
            return

        print(seq_str)

    def add(self, method_name, *args, **kwargs):
        """
        This function adds a drone command to be scheduled in the sequence.

        :param method_name: string value, name of Drone function (command) to be scheduled
        :param *args: value(s) for the parameter of the given Drone function.
        :param **kwargs: value(s) for the parameter of the given Drone function, in form of parameter=value.
        """
        self._sequence[self.index].append([method_name, args, kwargs])

    def get_sequence(self):
        return self._sequence

## Sequence Class End ##
