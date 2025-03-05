import os
import json

from gcsnap.rich_console import RichConsole
from gcsnap.configuration import Configuration
from gcsnap.genomic_context import GenomicContext
from gcsnap.db_handler_sequences import SequenceDBHandler
from gcsnap.parallel_tools import ParallelTools
from gcsnap.utils import split_dict_chunks


class Sequences:
    """ 
    Methods and attributes to get the sequences for the flanking genes of the target genes.
    They are stored in a structure like:
        data-path as defined in config.yaml or via CLI
        ├── genbank
        │   └── data
        │       └── GCA_000001405.15_genomic.gff.gz
        ├── refseq
        │   └── data
        │       └── GCF_000001405.38_genomic.gff.gz
        ├── db
        │   └── assemblies.db
        │   └── mappings.db
        │   └── sequences.db
        │   └── rankedlineage.dmp

    Attributes:
        database_path (str): The path to the database.
        gc (GenomicContext): The GenomicContext object containing all genomic context information.
        chunks (int): The number of chunks to split the syntenies.
        sequences_dict (dict): The dictionary with the sequences of the flanking genes.
        console (RichConsole): The RichConsole object to print messages.
        sequences (dict): The results dictionary with the flanking genes and their sequences.
    """

    def __init__(self, config: Configuration, gc: GenomicContext):
        """
        Initialize the Sequences object.

        Args:
            config (Configuration): The Configuration object containing the arguments.
            gc (GenomicContext): The GenomicContext object containing all genomic context information.
        """        
        self.config = config
        # get necessary configuration arguments        
        self.database_path = os.path.join(config.arguments['data_path']['value'],'db') 
        self.chunks = (config.arguments['n_nodes']['value'] * config.arguments['n_ranks_per_node']['value'])
        self.gc = gc

        self.console = RichConsole()

    def get_sequences(self) -> dict:
        """
        Getter for the sequences attribute.

        Returns:
            dict: The dictionary with flanking genes and their sequences.
        """        
        return self.sequences        

    def run(self) -> None:
        """
        Run the assignment of sequences to the flanking genes.
            - Find sequences for all flanking genes.
            - Add sequences, tax id and species name to flanking genes.
        Uses parallel processing with the parallel_wrapper from ParallelTools.
        """        
        # Get syntenies
        syntenies = self.gc.get_syntenies()

        # extract target and neeeded flanking genes ncbi codes
        targets_ncbi_codes = {k: v['flanking_genes']['ncbi_codes'] for k, v in syntenies.items()}
        # split into chunks
        parallel_args = split_dict_chunks(targets_ncbi_codes, self.chunks)

        with self.console.status('Add sequences, tax id and species name to flanking genes'):
            dict_list = ParallelTools.parallel_wrapper(parallel_args, self.run_each)
            # combine results
            self.sequences = {k: v for d in dict_list for k, v in d.items()}

    def run_each(self, args: list[dict]) -> dict:
        """
        Run the assignment of sequences to the flanking genes for one target used
        in parallel processing.

        Args:
            args (tuple[dict]): The arguments for the sequence assignment.
                The dictionary with the flanking gene information

        Returns:
            dict: The dictionary with the flanking genes and their sequences.
        """        
        targets_ncbi_codes = args

        # get all ncbi codes
        ncbi_codes = [ncbi_code for content_dict in targets_ncbi_codes.values() for ncbi_code in content_dict]

        # get from database
        self.find_sequences(ncbi_codes)

        results = {}
        # adapt syntenies
        for target, values in targets_ncbi_codes.items():
            # update flanking genes with sequence
            # update flanking genes with sequence
            sequence_list = [self.get_sequence(ncbi_code) for ncbi_code in values]
            results |= {target: {'flanking_genes': {'sequences': sequence_list}}}
  
        return results

    def find_sequences(self, ncbi_codes: list) -> None:
        """
        Find sequences for all flanking genes using the EntrezQuery class.

        Args:
            ncbi_codes (list): The list of NCBI codes to find sequences for.
        """        
        # select from database
        sequences_db = SequenceDBHandler(os.path.join(self.database_path))
        self.sequences_dict = sequences_db.select_as_dict(ncbi_codes)

    def get_sequence(self, ncbi_code: str) -> str:
        """
        Get the sequence for a ncbi code. 
        If the sequence is not found, a fake sequence is returned.

        Args:
            ncbi_code (str): The NCBI code.

        Returns:
            str: The sequence for the NCBI code.
        """        
        return self.sequences_dict.get(ncbi_code, 'FAKESEQUENCEFAKESEQUENCEFAKESEQUENCEFAKESEQUENCE')        


    