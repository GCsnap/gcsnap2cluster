"""
This script is used to update the sequence databases.
The original db_create_assemblies.py scriptwas using a wrong implementation
of dh_handler_sequences.py. This was correct but the sequences need to be updated.
The Primary Key of the sequences table is the ncbi_code, so if the same ncbi_code.
The indexes are already created, so we just update the seqeucences.
"""

import os
import sys
import glob
import time
import datetime

# import database handlers
from gcsnap.db_handler_sequences import SequenceDBHandler
from gcsnap.utils import processpool_wrapper
from gcsnap.utils import split_list_chunks_size
      
        
def split_into_parts(lst: list, n: int) -> list[list]:
    # k, how many times n fits in len(list), m the reminder
    q, r = divmod(len(lst), n)
    return [lst[i * q + min(i, r):(i + 1) * q + min(i + 1, r)] for i in range(n)]
    

def execute_handler(args: tuple):
    handler, batch = args
    # each handler returns a list of tuples
    sequences, _ = handler.parse_sequences_from_faa_files(batch)
    return sequences


def execute_handlers(handler: SequenceDBHandler, batch: list, n_processes: int) -> tuple[list[tuple[str,str]], list[tuple[str,str]]]:
    # split the batch in equal sized parts
    batches = split_into_parts(batch, n_processes) 
    parallel_args = [(handler, subbatch) for subbatch in batches]
    
    # retrurns a list of tuples containing lists of tuples
    result = processpool_wrapper(n_processes, parallel_args, execute_handler)
    
    # flatten the list of tuples containing lists of tuples
    #mappings = [item for sublist in result for item in sublist[1]]
    sequences = [item for sublist in result for item in sublist]
    
    return sequences

def update_dbs(path: str, n_processes: int) -> None:    

    print('{}: Start updating databases'.format(time.ctime(time.time())), flush=True)

    # where to create the databases
    db_dir = os.path.join(path, 'db')           
    # open assembly database handler and create tables
    seq_db_handler = SequenceDBHandler(db_dir)   
    
    # number of files to write in parallel
    batch_size = n_processes * 10
    # keep track of sequences and assemblies
    n_assemblies = 0 
    n_sequences = 0
        
    #for loop_var in ['genbank','refseq']:    
                 
        #db_type = loop_var

    # 1. Extract information from .faa files       
    # a) (ncbi_code, sequence) go into sepearte databases (as this will get huge)

    # Directory containing the .ffa files
    #faa_data_dir = os.path.join(path, db_type, 'data')
    # list of all files to parse      
    #file_paths = glob.glob(os.path.join(faa_data_dir,'*_protein.faa.gz'))

    # read needed assembly files for experiments: file was created handisch with assemblies_for_cluster.ipynb
    with open('/users/stud/k/kruret00/PASC25/targets/assemblies.txt', 'r') as file:
        content = file.read()
    lines = content.splitlines()
    file_names = [line.strip() + '_protein.faa.gz' for line in lines]
    # file_names = file_names[:1000]

    # add path structer
    file_paths =  [os.path.join(path, 'refseq', 'data', file_name) for file_name in file_names if file_name.startswith('GCF')]
    file_paths += [os.path.join(path, 'genbank', 'data', file_name) for file_name in file_names if file_name.startswith('GCA')]
            
    # we loop over all those files in batches, each database takes 
    # indexing is switched off to speed up
    batch_nr = 1
    batches = split_list_chunks_size(file_paths, batch_size)
    for batch in batches:

        print('{}: Do batch {} of {}'.format(time.ctime(time.time()), batch_nr, len(batches)), flush=True)
        
        # start the parsing for each handler, in parallel 
        sequence_list = execute_handlers(seq_db_handler, batch, n_processes)

        print('{}:   Files parsed'.format(time.ctime(time.time())), flush=True)
                        
        # add sequences
        seq_db_handler.disable_indices()
        seq_db_handler.batch_update_sequences(sequence_list)      
                    
        # Format numbers with thousand separators
        print('{}:   {:,} sequences with size {:.2f} MB updated'.format(time.ctime(time.time()), len(sequence_list), sum(len(seq) for _, seq in sequence_list) / (1024 * 1024)), flush=True)    
        
        # keep track of done things
        n_sequences += len(sequence_list)
        n_assemblies += len(batch)
        batch_nr += 1      

        print('{}:   {:,} Assemblies and {:,} sequences in total'.format(time.ctime(time.time()), n_assemblies, n_sequences), flush=True) 
            
    print('{}: All Updating done'.format(time.ctime(time.time())),flush=True)
    
    # 2. Rebuild indices
    print('{}: Start rebuilding indices'.format(time.ctime(time.time())),flush=True)
    seq_db_handler.reindex_sequences

    print('{}: Start querying number of entries'.format(time.ctime(time.time())),flush=True)
    # get number of unique sequences
    n_unique = seq_db_handler.select_number_of_entries()

    return n_assemblies, n_sequences, n_unique                   

# Example usage
if __name__ == "__main__":    
   
    #n_processes = int(sys.argv[1])
    n_processes = 20

    path = '/storage/shared/msc/gcsnap_data/'
           
    st = time.time()
    n_assemblies, n_sequences, n_unique = update_dbs(path, n_processes)
    
    elapsed_time = time.time() - st
    formatted_time = str(datetime.timedelta(seconds=round(elapsed_time)))

    # Format numbers with thousand separators
    formatted_assemblies = "{:,}".format(n_assemblies)
    formatted_sequences = "{:,}".format(n_sequences)
    formatted_unique = "{:,}".format(n_unique)
    
    print('{}: {} assemblies with {} sequences ({} unique sequences found) done in {}'.
          format(time.ctime(time.time()), formatted_assemblies, formatted_sequences, formatted_unique, formatted_time))

    print('Done')
 