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

# Split dictionary into list of dictoonary chunks
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

# ------------------------------------------------------
""" 
Various parallelization methods to apply a function to a list of argument tuples in parallel.
including:
    - sequential_wrapper: Apply a function to a list of arguments sequentially.
    - threading_wrapper: Apply a function to a list of arguments using threads.
    - processpool_wrapper: Apply a function to a list of arguments using a process pool.
    - threadpool_wrapper: Apply a function to a list of arguments using a thread pool.
    - futures_thread_wrapper: Apply a function to a list of arguments using ThreadPoolExecutor.
    - futures_process_wrapper: Apply a function to a list of arguments using ProcessPoolExecutor.
"""

from typing import Callable

from threading import Thread, Lock

from multiprocessing import Pool as ProcessPool
from multiprocessing.pool import ThreadPool

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed

import os
# supress warning about depracted omp command
# OMP: Info #276: omp_set_nested routine deprecated, please use omp_set_max_active_levels instead
# the reasons for this remains unclear, as omp should be up to date as we added gcc=14.1
# it is for sure caused by multiprocessing, but it is not a problem for the code
# Moreover, it only shows on certain macOS systems, with the new AMD M1 chip
# but we won't to avoid the warning as useres can't do anything about it
os.environ['OMP_DISPLAY_ENV'] = 'FALSE'

def sequential_wrapper(parallel_args: list[tuple], func: Callable) -> list:
    """
    Apply a function to a list of arguments sequentially. The arguments are passed as tuples
    and are unpacked within the function.

    Args:
        parallel_args (list[tuple]): A list of tuples, where each tuple contains the arguments for the function.
        func (Callable): The function to apply to the arguments.

    Returns:
        list: A list of results from the function applied to the arguments.
    """        
    result_list = []
    
    for arg in parallel_args:
        result_list.append(func(arg))   
        
    return result_list

def threading_wrapper(parallel_args: list[tuple], func: Callable) -> list:
    """
    Apply a function to a list of arguments using threads. The arguments are passed as tuples
    and are unpacked within the function.

    Args:
        parallel_args (list[tuple]): A list of tuples, where each tuple contains the arguments for the function.
        func (Callable): The function to apply to the arguments.

    Returns:
        list: A list of results from the function applied to the arguments.
    """    
    def worker(index):
        result = func(parallel_args[index])
        with lock:
            results_list[index] = result

    results_list = [None] * len(parallel_args)
    threads = []
    lock = Lock()

    for i in range(len(parallel_args)):
        t = Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    return results_list

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

def threadpool_wrapper(n_threads: int, parallel_args: list[tuple], func: Callable) -> list:
    """
    Apply a function to a list of arguments using a thread pool. The arguments are passed as tuples
    and are unpacked within the function, meaning the function takes only one argument, but a tuple.
    This allows the use of the map function of the pool which takes only one argument.
    They are asynchronus, meaning the order of the results is not guaranteed.

    Args:
        n_threads (int): The number of threads to use.
        parallel_args (list[tuple]): A list of tuples, where each tuple contains the arguments for the function.
        func (Callable): The function to apply to the arguments.

    Returns:
        list: A list of results from the function applied to the arguments in the order they finish.
    """    
    pool = ThreadPool(n_threads)

    # Use map_async to apply the function asynchronously
    results = pool.map_async(func, parallel_args)

    pool.close() # close pool for further processes
    pool.join() # wait until all have finished.

    # Convert to a list of results from the MapResult object
    result_list = results.get()
    return result_list

def futures_thread_wrapper(n_threads: int, parallel_args: list[tuple], func: Callable) -> list:
    """
    Apply a function to a list of arguments using ThreadPoolExecutor. The arguments are passed as tuples
    and are unpacked within the function. As completed is used to get the results in the order they finish.

    Args:
        n_threads (int): The number of threads to use.
        parallel_args (list[tuple]): A list of tuples, where each tuple contains the arguments for the function.
        func (Callable): The function to apply to the arguments.

    Returns:
        list: A list of results from the function applied to the arguments in the order they finish.
    """    
    # executor.submit() to submit each task to the executor. 
    # This returns a Future object for each task. You then use as_completed() 
    # to get an iterator that yields futures as they complete. 
    # Create a ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = [executor.submit(func, arg) for arg in parallel_args]
        # list comprehension with future.result() to get the results of the futures. 
        # This means that your code will start processing the results as soon as they become available, in the order they finish.
        result_list = [future.result() for future in as_completed(futures)]
        # but it will wait till all are done to go on

    return result_list

def futures_process_wrapper(n_processes: int, parallel_args: list[tuple], func: Callable) -> list:
    """
    Apply a function to a list of arguments using ProcessPoolExecutor. The arguments are passed as tuples
    and are unpacked within the function. As completed is used to get the results in the order they finish.

    Args:
        n_processes (int): The number of processes to use.
        parallel_args (list[tuple]): A list of tuples, where each tuple contains the arguments for the function.
        func (Callable): The function to apply to the arguments.

    Returns:
        list: A list of results from the function applied to the arguments in the order they finish.
    """    
    # Same as with futures and threads
    # Create a ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=n_processes) as executor:
        futures = [executor.submit(func, arg) for arg in parallel_args]
        result_list = [future.result() for future in as_completed(futures)]

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

# loggin never done in parallel:
# The reason is that logging from several processes is not that easy
# https://docs.python.org/3/howto/logging-cookbook.html#logging-to-a-single-file-from-multiple-processes

class CustomFormatter(logging.Formatter):
    """
    Custom formatter for logging messages.

    Attributes:
        message (str): The message to log.
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

logging.basicConfig(
    filename='gcsnap.log',
    filemode='w',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Create a custom formatter
formatter = CustomFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Set a higher logging level for specific loggers
logging.getLogger('asyncio').setLevel(logging.WARNING)
logger = logging.getLogger()

# Update handlers to use the custom formatter
for handler in logger.handlers:
    handler.setFormatter(formatter)
# ------------------------------------------------------

