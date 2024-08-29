# Exception classes
# ------------------------------------------------------
class WarningToLog(Exception):    
    """
    Exception to raise when a warning should be logged.

    Attributes:
        message (str): The message to log.
    """

    def __init__(self, message: str):
        """
        Initialize the exception.

        Args:
            message (str): The message to log.
        """        
        self.message = message
        super().__init__(self.message)       
# ------------------------------------------------------      

# Split dictionary into list of dictionary chunks
# ------------------------------------------------------
def split_dict_chunks(input_dict: dict, n_chunks: int) -> list[dict]:
    """
    Split a dictionary into n_chunks sub-dictionaries.

    Args:
        input_dict (dict): The dictionary to split.
        n_chunks (int): The number of sub-dictionaries to create.

    Returns:
        list[dict]: A list of n_chunks sub-dictionaries.
    """    
    # list of all key-value pairs, a list of tuples
    key_values = list(input_dict.items())  
    sub_lists = split_list_chunks(key_values, n_chunks)

    # back to dictionary
    return [dict(sub_list) for sub_list in sub_lists]
# ------------------------------------------------------

# Split list into list of chunks
# ------------------------------------------------------
def split_list_chunks(input_list: list, n_chunks: int) -> list[list]:
    """
    Split a list into n_chunks sub-lists.

    Args:
        input_list (list): The list to split.
        n_chunks (int): The number of sub-lists to create.

    Returns:
        list[list]: A list of n_chunks sub-lists.
    """    
    n_values = len(input_list)
    # needs some addition take care as the last part might be empty
    # like for 100 targets with 16 chunks, the step is 100//16+1=7 and 15*7>100
    # in such a case we use 100//16=6 and we make last batch larger than the previous ones        
    incrementation = 1 if (n_values // n_chunks) * (n_chunks-1) >= n_values else 0 
    n_each_list = (n_values // n_chunks) + incrementation
    # create cores-1 sub lists equally sized
    sub_lists = [input_list[((i-1)*n_each_list):(i*n_each_list)]
                    for i in range(1, n_chunks)]
    # the last will just have all the remaining values
    sub_lists = sub_lists + [input_list[((n_chunks-1)*n_each_list):]] 

    return sub_lists
# ------------------------------------------------------

# Split list into chunks of at most size limit
# ------------------------------------------------------
def split_list_chunks_size(ids: list, size_limit: int = 1000) -> list:
    """
    Split the IDs into sublists of size less than or equal to the API limit.

    Args:
        ids (list): The list of IDs to split.
        size_limit (int): The limit for the size of each sublist. Defaults to 1000.

    Returns:
        list: The list of sublists of IDs.
    """
    # Initialize an empty list to store the chunks
    chunks = []

    # Loop through the list in increments of api_limit
    for i in range(0, len(ids), size_limit):
        # Append a sublist of the current chunk to the chunks list
        chunks.append(ids[i:i + size_limit])

    return chunks
# ------------------------------------------------------


# Split dict into chunks of at most size limit
# ------------------------------------------------------
def split_dict_chunks_size(data: dict, size_limit: int = 1000) -> list:
    """
    Split the dictionary into sublists of dictionaries, each with a size
    less than or equal to the size limit.

    Args:
        data (dict): The dictionary to split.
        size_limit (int): The limit for the size of each sub-dictionary. Defaults to 1000.

    Returns:
        list: A list of dictionaries, each containing up to size_limit key-value pairs.
    """
    # Initialize an empty list to store the chunks
    chunks = []

    # Convert the dictionary items to a list of tuples
    items = list(data.items())

    # Loop through the list of items in increments of size_limit
    for i in range(0, len(items), size_limit):
        # Create a new dictionary for the current chunk and add it to chunks
        chunk = dict(items[i:i + size_limit])
        chunks.append(chunk)

    return chunks
# ------------------------------------------------------

""" 
Various parallelization methods to apply a function to a list of argument tuples in parallel.
including:
    - processpool_wrapper: Apply a function to a list of arguments using a process pool.
"""

from typing import Callable
from multiprocessing import Pool as ProcessPool

import os
# supress warning about depracted omp command
# OMP: Info #276: omp_set_nested routine deprecated, please use omp_set_max_active_levels instead
# the reasons for this remains unclear, as omp should be up to date as we added gcc=14.1
# it is for sure caused by multiprocessing, but it is not a problem for the code
# Moreover, it only shows on certain macOS systems, with the new AMD M1 chip
# but we won't to avoid the warning as useres can't do anything about it
os.environ['OMP_DISPLAY_ENV'] = 'FALSE'

def processpool_wrapper(n_processes: int, parallel_args: list[tuple], func: Callable) -> list:       
    """
    Apply a function to a list of arguments using a process pool. The arguments are passed as tuples
    and are unpacked within the function, meaning the function takes only one argument, but a tuple.
    This allows the use of the map function of the pool which takes only one argument.
    They are asynchronus, meaning the order of the results is not guaranteed.

    Args:
        n_processes (int): The number of processes to use.
        parallel_args (list[tuple]): A list of tuples, where each tuple contains the arguments for the function.
        func (Callable): The function to apply to the arguments.

    Returns:
        list: A list of results from the function applied to the arguments in the order they finish.
    """        
    pool = ProcessPool(processes = n_processes)
    
    # imap and map take just one argument, hence unpacking within function
    # imap is the lazy version of map,
    # _unordered is without controlling the result order. (map_async)
    # multiple arguments use startmap(),
    # but starmap() and map() may be inefficient with large lists:
        # https://docs.python.org/3/library/multiprocessing.html#multiprocessing.pool.Pool.map
    # great explanation about:
        # https://stackoverflow.com/questions/26520781/multiprocessing-pool-whats-the-difference-between-map-async-and-imap
        
    results = pool.map_async(func, parallel_args)
    
    pool.close() # close pool for further processes
    pool.join() # wait until all have finished.
    
    result_list = results.get()

    return result_list
# ------------------------------------------------------


# ------------------------------------------------------
""" 
Logging within the GCsnap pipeline.

Log desired information to gcnap.log file. The log file is created in the working directory.
Logging levels are set to INFO by default but set to WARNING for the asyncio logger.
Loggin messages are formatted as follows:
    - Timestamp
    - Logger name
    - Log level
    - Log message   
"""
import logging # already sets the loggin process

# logging never done in parallel:
# The reason is that logging from several processes is not that easy
# https://docs.python.org/3/howto/logging-cookbook.html#logging-to-a-single-file-from-multiple-processes

class CustomFormatter(logging.Formatter):
    """
    Custom formatter for logging messages.
    """    
    def format(self, record: str) -> str:
        """
        Initialize the formatter.

        Args:
            record (str): The format of the log record.

        Returns:
            str: The formatted log record.
        """        
        record.name = record.name.split('.')[-1]
        return super().format(record)
    
class CustomLogger():    
    """
    Custom logger for the GCsnap pipeline.
    There are two loggers:
        - 'base': The base logger for the entire pipeline.
        - 'iteration': The logger for a specific task, as there can be multiple 
                        tasks in one run.
    """ 
    # Configure the initial logger for steps 1 and 2

    @classmethod
    def configure_loggers(cls) -> None:
        """
        Configure the base and iteration loggers.
        """

        # Base logger configuration
        logger_base = logging.getLogger('base')
        logger_base.setLevel(logging.INFO)

        # Define the new log file name for the current iteration
        base_log_file = os.path.join(os.getcwd(), 'gcsnap.log')

        base_handler = logging.FileHandler(base_log_file, mode = 'w')  # Base log file
        formatter = CustomFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        base_handler.setFormatter(formatter)
        logger_base.addHandler(base_handler)

        # Iteration logger configuration
        logger_iteration = logging.getLogger('iteration')
        logger_iteration.setLevel(logging.INFO)

        # We won't set the iteration handler here; it should be set in the context of its use
        # so each iteration can have a different file or configuration as needed.

        # Set a higher logging level for specific loggers if needed
        logging.getLogger('asyncio').setLevel(logging.WARNING)

    @classmethod
    def configure_iteration_logger(cls, out_label: str, starting_directory: str) -> None:
        """
        Configure the iteration logger for a specific iteration.

        Args:
            out_label (str): The label of the task equal to the folder in which the output is stored.
            starting_directory (str): The starting directory of the pipeline.
        """
        logger_iteration = logging.getLogger('iteration')

        # Define the new log file name for the current iteration
        iteration_log_file = os.path.join(os.getcwd(), f'gcsnap_{out_label}.log')
        base_log_file = os.path.join(starting_directory, 'gcsnap.log')

        # Copy the base log content to the iteration log file if needed
        cls.copy_log_content(base_log_file, iteration_log_file)

        # Remove existing iteration handlers
        for handler in logger_iteration.handlers[:]:
            logger_iteration.removeHandler(handler)

        # Add a new handler for the iteration log file
        iteration_handler = logging.FileHandler(iteration_log_file, mode = 'a')
        formatter = CustomFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        iteration_handler.setFormatter(formatter)
        logger_iteration.addHandler(iteration_handler)

    # Copy the log file content to ensure all data is saved
    @classmethod
    def copy_log_content(cls, base_log_file: str, iteration_log_file: str) -> None:
        """
        Copy the contents of the base log file to the iteration log file.

        Args:
            base_log_file (str): The path to the base log file.
            iteration_log_file (str): The path to the iteration log file.
        """        
        # Read the contents of the base log file
        with open(base_log_file, 'r') as base_log:
            log_content = base_log.read()
        # Write the contents to the iteration log file
        with open(iteration_log_file, 'w') as iter_log:
            iter_log.write(log_content)

    @classmethod
    def log_to_base(cls, msg: str) -> None:
        """
        Log a message to the base logger.

        Args:
            msg (str): The message to log.
        """        
        logger = logging.getLogger('base')
        logger.info(msg)

    @classmethod
    def log_to_iteration(cls, msg: str) -> None:
        """
        Log a message to the iteration logger.

        Args:
            msg (str): The message to log.
        """        
        logger = logging.getLogger('iteration')
        logger.info(msg)        
# ------------------------------------------------------
