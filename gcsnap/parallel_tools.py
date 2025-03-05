
from typing import Callable

from concurrent.futures import as_completed
from mpi4py.futures import MPIPoolExecutor

from gcsnap.configuration import Configuration
from gcsnap.rich_console import RichConsole 

import time
from datetime import datetime
import os
import inspect

class ParallelTools:

    """
    Methods and attributes to handle parallel processing. The class is initialized with a Configuration object.

    Attributes:
        config (Configuration): The Configuration object with the parsed arguments.
        n_nodes (int): The number of nodes to use for parallel processing.
        n_cpu (int): The number of CPUs per node to use for parallel processing.
        workers (int): The number of workers to use for parallel processing with mpoi4py.
        memory_per_node (int): The memory per node to use for parallel processing.
        console (RichConsole): The RichConsole object.
    """

    def __init__(self, config: Configuration):
        """
        Initialize the ParallelTools object.

        Args:
            config (Configuration): The Configuration object with the parsed arguments.
        """        
        self.config = config
        self.n_nodes = config.arguments['n_nodes']['value']
        self.n_cpu = config.arguments['n_ranks_per_node']['value']
        self.cluster = None

         # for mpi
        self.workers = (self.n_nodes * self.n_cpu)
            
        self.console = RichConsole()

        # Assign this instance to the class-level variable
        # Now self is available in the static method as on class level
        # usually instance is not on class level
        ParallelTools._instance = self

    @staticmethod
    def parallel_wrapper(parallel_args: list[tuple], func: Callable) -> list:
        """
        A static method that calls the process_wrapper method of the stored instance.
        This is called from the modules using parallel processing.

        Args:
            parallel_args (list[tuple]):A list of tuples, where each tuple contains the arguments for the function.
            func (Callable): The function to apply to the arguments.

        Returns:
            list: A list of results from the function applied to the arguments in the order they are provided.            
        """
        if ParallelTools._instance is None:
            raise RuntimeError("ParallelTools instance has not been initialized.")
        
        return ParallelTools._instance.mpiprocess_wrapper(parallel_args, func)       

    def mpiprocess_wrapper(self, parallel_args: list[tuple], func: Callable) -> list:
        """
        Apply a function to a list of arguments using ProcessPoolExecutor. The arguments are passed as tuples
        and are unpacked within the function. As completed is used to get the results in the order they finish.

        Args:
            parallel_args (list[tuple]): A list of tuples, where each tuple contains the arguments for the function.
            func (Callable): The function to apply to the arguments.

        Returns:
            list: A list of results from the function applied to the arguments in the order they finish.
        """            
        # Same as with ProcessPoolExecutor from cuncurrent.futures
        # https://mpi4py.readthedocs.io/en/stable/mpi4py.futures.html#parallel-tasks
        # with MPIPoolExecutor(max_workers = workers) as executor:
        with MPIPoolExecutor(max_workers = self.workers) as executor:        

            # build parallel args including func and length of parallel_args list
            parallel_args_2 = [(func, arg, len(parallel_args), executor.num_workers) for arg in parallel_args]

            # futures = [executor.submit(func, arg) for arg in parallel_args]
            futures = [executor.submit(self.timed_func, arg) for arg in parallel_args_2]
            result_list = [future.result() for future in as_completed(futures)]

        return result_list
    
    # helper function to time func
    def timed_func(self, args):
        """
        Function to collect MPI rank information and write timing information to a log file.

        Args:
            args (tuple): A tuple containing the function, the arguments, and the length of the arguments list.

        Returns:
            list: The result of the function applied to the arguments.
        """
        # Get timestamp maily for sorting the output
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # execute function
        func, arg, all_len, workers = args
        startstamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        start_t = time.time()
        result = func(arg)
        end_t = time.time()
        endstamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # create arg lenght summary
        num_args = len(arg)
        size_args = []
        type_args = []
        for e in arg:
            type_args.append(type(e))
            if type(e) == str or type(e) == bool:
                size_args.append(1)
            else:
                size_args.append(len(e))
        args_s = f'Num Args: {num_args}, Type Args: {type_args}, Size Args: {size_args}'

        # duration
        dur_t = end_t - start_t

        # Identify function name
        if hasattr(func, '__name__'):  # Regular function
            func_name = func.__name__
        elif hasattr(func, '__class__') and hasattr(func, '__call__'):  # Class instance with __call__
            func_name = func.__class__.__name__
        else:
            func_name = 'unknown_function'

        # Get rank if running in MPI environment
        rank = None
        if "OMPI_COMM_WORLD_RANK" in os.environ:
            rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
        elif "PMI_RANK" in os.environ:
            rank = int(os.environ["PMI_RANK"])
        elif os.getenv("SLURM_PROCID") is not None:
            rank = int(os.getenv("SLURM_PROCID"))
        else:
            rank = os.getpid()  # Fallback to process ID

        # path hard coded
        log_path = '/users/stud/k/kruret00/PASC25/experiments_deliver/MPIprofiling/rank_results'

        # Log timing information (append to the same file per worker)
        log_file = f'func_{func_name}_mpi_worker_{rank}.log'

        # strings to write
        func_s = f'Func: {func_name}'
        all_len_s = f'Len: {all_len}'
        workers_s = f'Workers: {workers}'
        rank_s = f'Rank: {rank}'
        pid_s = f'PID: {os.getpid()}'
        start_s = f'Start: {startstamp}'
        end_s = f'End: {endstamp}'
        dur_s = f'Duration: {dur_t:.6f}'

        # write to file
        with open(os.path.join(log_path,log_file), 'a') as f:
            f.write(f'{timestamp}, {func_s}, {all_len_s}, {workers_s}, {rank_s}, {pid_s}, {args_s}, {start_s}, {end_s}, {dur_s}\n')

        # return result and time
        return result