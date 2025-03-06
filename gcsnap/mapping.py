import pandas as pd
import numpy as np
import os

from gcsnap.configuration import Configuration
from gcsnap.rich_console import RichConsole
from gcsnap.parallel_tools import ParallelTools
from gcsnap.uniprot_dbs_dict import UniprotDict
from gcsnap.db_handler_uniprot_mappings import UniprotMappingsDBHandler

import logging
logger = logging.getLogger('iteration')

class SequenceMapping:
    """ 
    Methods and attributes to extract the target sequences from the mapping database.
    The resulting DataFrame is the base to map targets to other types and also
    retrieve the Taxonomy ID and the PDB ID.
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
        config (Configuration): The Configuration object containing the arguments.
        target_list (list): The list of target sequences.
        data_path (str): The path to the data directory.
        console (RichConsole): The RichConsole object to print messages.
        uniprot_dict (dict): The dictionary with the identifiers standard information and the target sequences.
        mapping_df (pd.DataFrame): The DataFrame with the mapping results.
        supported (list): The list of supported databases.
        target_types (list): The list of target types.
    """

    def __init__(self, config: Configuration, target_list: list, msg: str = ''):
        """
        Initialize the SequenceMapping object.

        Args:
            config (Configuration): The Configuration object containing the arguments.
            path_to_raw_mapping (str): The path to the raw mapping file including the file name.
            target_list (list, optional): The list of target sequences. Defaults to None
            block_size (int, optional): The block size of the partition files to create. Defaults to 128e6.
        """        
        self.target_list = target_list
        self.config = config
        self.msg = msg

        # extract needed values from config
        self.database_path = os.path.join(config.arguments['data_path']['value'],'db')        
        self.console = RichConsole()

    def run(self):
        """
        Run the mapping of target sequences to the specified id standard:
            - Create the UniProt dictionary with the target sequences.
            - Create the parallel input for the mapping.
            - Run the mapping.
            - Finalize the mapping.
        Uses parallel processing with the parallel_wrapper from ParallelTools. 
        """        
        with self.console.status('Mapping {}'.format(self.msg)):
            self.create_uniprot_dbs_with_targets()
            parallel_args = self.create_parallel_input()
            mapping_df_list = ParallelTools.parallel_wrapper(parallel_args, self.do_mapping)
            # concatenate all mapping dfs to one
            self.mapping_df = pd.concat(mapping_df_list, ignore_index=True)
            # add target column
            self.finalize()

    def run_fam(self):
        """
        Run the mapping of target sequences to the specified id standard used for profiling for family function annotation
            - Create the UniProt dictionary with the target sequences.
            - Create the parallel input for the mapping.
            - Run the mapping.
            - Finalize the mapping.
        Uses parallel processing with the parallel_wrapper from ParallelTools. 
        """        
        with self.console.status('Mapping {}'.format(self.msg)):
            self.create_uniprot_dbs_with_targets()
            parallel_args = self.create_parallel_input()
            mapping_df_list = ParallelTools.parallel_wrapper(parallel_args, self.do_mapping_fam)
            # concatenate all mapping dfs to one
            self.mapping_df = pd.concat(mapping_df_list, ignore_index=True)
            # add target column
            self.finalize()            

    def finalize(self): 
        """
        Finalize the mapping of target sequences to the specified database:
            - Add the NCBI column.
            - Log the failed mappings.
            - Save the mapping DataFrame to a CSV file.        
        """        
        # possible existing target column is overwritten
        self.add_target_column()
        self.add_ncbi_column()
        self.log_failed_finalize()

        # write dataframe to csv file
        self.mapping_df.to_csv('mapping.csv', index=False) 
        self.console.print_done('All mapping done. Table saved to mapping.csv')        
    
    def get_codes(self, id_type: str) -> list:
        """
        Get the list of codes from the mapping DataFrame.

        Args:
            id_type (str, optional): The id standard of the codes to get. Defaults to None using to_type attribute.

        Returns:
            list: The list of codes.
        """           
        return self.mapping_df[id_type].dropna().tolist()
    
    def get_targets_and_ncbi_codes(self) -> list[tuple]:
        """
        Get a list of tuples with the target sequences and the NCBI codes.

        Returns:
            list[tuple]: The list of tuples with the target sequences and the NCBI codes.
        """        
        # return list of tuples with target and ncbi_code
        df = self.mapping_df[(self.mapping_df['target'].notna()) &
                              (self.mapping_df['ncbi_code'].notna())]        
        return list(zip(df['target'], df['ncbi_code']))
    
    def get_target_to_result_dict(self, id_type: str) -> dict:
        """
        Get a dictionary with the target sequences as keys and the mapped results as values.

        Returns:
            dict: The dictionary with the target sequences and the mapped results.
        """        
        df = self.mapping_df[(self.mapping_df['target'].notna()) &
                              (self.mapping_df[id_type].notna())] 
        return df.set_index('target')[id_type].to_dict()    
           
    def create_uniprot_dbs_with_targets(self) -> None:
        """
        Create the UniProt dictionary and assign the target sequences to the different type standards.
        """        
        uniprot_dict = UniprotDict()
        # fill uniprot_dbs with the targets        
        uniprot_dict.assign_target_to_type(self.target_list)
        self.supported = uniprot_dict.get_supported_databases()
        self.target_types = uniprot_dict.get_target_types()
        self.uniprot_dict = uniprot_dict.get_uniprot_dict()

    def create_parallel_input(self, api_limit: int = 1000) -> list[tuple]:
        """
        Create the parallel input for the mapping of target sequences from the
        correct database column.

        Returns:
            list[tuple]: The list of tuples with the parallel input arguments.
        """        
        parallel_input_args = []

        # 1. Create a parallel workload for each uniprot type
        # This assures that we can do all of the same type with one query
        for key, values in self.uniprot_dict.items():
            if len(values['targets']) != 0:
                # we used underscores in the database names, but the keys are with dashes
                key = key.replace('-', '_')
                ids = values['targets']
                
                parallel_input_args.append((ids, key))

        # 2. Split the ids into sublists of more than api_limit
        # This assures that we do not exceed the api limit
        refinded_parallel_input_args = []
        additions = []
        ids_sublists = []
        for parallel_arg in parallel_input_args:
            ids , key = parallel_arg    

            if len(ids) > api_limit:
                ids_sublists += self.split_recursive(ids, api_limit)
                additions += [(sublist, key) for sublist in ids_sublists]
            else:
                refinded_parallel_input_args.append(parallel_arg)

        parallel_input_args = refinded_parallel_input_args + additions
        return parallel_input_args
    
    def split_recursive(self, ids: list, api_limit: int) -> list:
        """
        Split the IDs into sublists of more than the API limit.

        Args:
            ids (list): The list of IDs to split.
            api_limit (int): The API limit for the number of IDs to map.

        Returns:
            list: The list of sublists of IDs.
        """        
        # Base case: if the length of ids is less than or equal to api_limit, 
        # return a list containing ids
        if len(ids) <= api_limit:
            # the trick is to return [ids], a list of a list which we can iterate through
            return [ids]
        
        # Recursive case: split ids into two halves and recursively split each half
        mid = len(ids) // 2
        return self.split_recursive(ids[:mid], api_limit) + self.split_recursive(ids[mid:], api_limit)  

    def do_mapping(self, arg: tuple[list,str]) -> pd.DataFrame:
        """
        Do the mapping of target sequences to the specified database used
        in the parallel processing:
            - Get mapping from DB
            - Make the mapping DataFrame unique.         

        Args:
            arg (tuple[list,str]): The tuple with the arguments.
                First element is the list of IDs to map.
                Second element is the field in the mapping database to query.

        Returns:
            pd.DataFrame: The DataFrame with the unique mapping results.
        """ 
        ids, field = arg
        mapping_df = self.query_mapping_db(ids, field) 
        unique_mapping_df = self.make_unique(mapping_df, [field])
        
        return unique_mapping_df
    
    def do_mapping_fam(self, arg: tuple[list,str]) -> pd.DataFrame:
        """
        Do the mapping of target sequences to the specified database used
        in the parallel processing for functional annotation to profile
            - Get mapping from DB
            - Make the mapping DataFrame unique.         

        Args:
            arg (tuple[list,str]): The tuple with the arguments.
                First element is the list of IDs to map.
                Second element is the field in the mapping database to query.

        Returns:
            pd.DataFrame: The DataFrame with the unique mapping results.
        """ 
        ids, field = arg
        mapping_df = self.query_mapping_db(ids, field) 
        unique_mapping_df = self.make_unique(mapping_df, [field])
        
        return unique_mapping_df

    def query_mapping_db(self, ids: list, field: str) -> pd.DataFrame:
        """
        Query the mapping database for the target sequences and the specified id standard.

        Args:
            ids (list): The list of target sequences.
            field (str): The database id standard to map from.

        Returns:
            pd.DataFrame: The DataFrame with the mapping results.
        """        
        # get mapping from database
        db_handler = UniprotMappingsDBHandler(self.database_path)
        # no return fields specified, so all fields are returned
        mapping_df = db_handler.fetch_records_as_dataframe(ids, field)
        return mapping_df

    def make_unique(self, mapping_df: pd.DataFrame, from_type: list) -> pd.DataFrame:
        """
        Make the mapping DataFrame unique by removing duplicates. If there is a tie of same duplicates,
        select the one with the shorter 'to_type' id.

        Args:
            mapping_df (pd.DataFrame): The DataFrame with the mapping results.
            from_type (list): The list of database id standards to map from.
            to_type (str): The database id standard to map to.

        Returns:
            pd.DataFrame: The DataFrame with the unique mapping results.
        """        
        # make copy before handling data frame
        df = mapping_df.copy()
        # tie criterion is shorter UniProtKB-AC
        to_type = 'UniProtKB_AC'

        # add length colum for sorting
        # Rule of thumb, just take the first one for UniParc.            
        # The longer UniProt-KB are currated automatically. Take the samller one
        # pd.NA is a float, hence len() directly is not possible
        df['Length'] = df[to_type].apply(lambda x: len(x) if pd.notna(x) else 0)
        # sort, default ascending
        df = df.sort_values(by='Length')         

        # do it for every column in from_type
        for col in from_type: 
            # is duplicated detects Nan as duplicated, avoid this
            mask = df[col].notna()
            # Checking for duplicates only on non-NaN values while preserving the original index
            is_duplicated = df.loc[mask, col].duplicated(keep=False).reindex(df.index, fill_value=False)
            # DataFrame of non-unique entries, selecting the first occurrence of each duplicated entry
            # sorting is not changed by group by
            first_non_unique_df = df[is_duplicated].groupby(col, sort=False).head(1)
            # DataFrame of unique entries (i.e., entries that are not duplicated)
            unique_df = df[~is_duplicated]
            # Combine the two DataFrames
            filtered_unique_df = pd.concat([unique_df, first_non_unique_df], ignore_index=True)
        
        # drop sorting column
        filtered_unique_df.drop(columns='Length', inplace=True)   
        return filtered_unique_df
    
    def add_target_column(self) -> None:
        """
        Add the target column to the mapping DataFrame and fill it with the target sequences.
        """        
        # add target column with NA or fill it with NA if present
        self.mapping_df['target'] = pd.NA
        for key in self.target_types:
            targets = self.uniprot_dict[key]['targets']
            key = key.replace('-', '_')            
            # Update the 'target' column with the value from 'key' column where 'key' is in targets
            self.mapping_df.loc[self.mapping_df[key].isin(targets), 'target'] = self.mapping_df[key]

    def add_ncbi_column(self) -> None:
        """
        Add the NCBI column to the mapping.
        """        
        self.mapping_df['ncbi_code'] = pd.NA

        # Update 'ncbi_code' with values from 'RefSeq' if not NA, otherwise from 'EMBL-CDS'
        # np.where(condition, x, y) returns x if condition is True, otherwise y
        self.mapping_df['ncbi_code'] = np.where(self.mapping_df['RefSeq'].notna(), 
                                                self.mapping_df['RefSeq'], self.mapping_df['EMBL_CDS'])

    def log_failed_finalize(self) -> None:
        """
        Log the failed mappings and finalize the mapping of target sequences to the specified database.
        """        
        df = self.mapping_df.copy()

        # those that were not found at all in the mapping
        targets = []
        for key in self.target_types:
            targets += self.uniprot_dict[key]['targets']
        missing = list(set(targets).difference(set(df['target'].to_list())))
        if len(missing) > 0:
            message = '{} ids is not containd in mapping database.'.format(len(missing))
            self.console.print_warning(message)
            for id in missing:
                logger.warning(f'Target sequence {id}') 

        # RefSeqs where no UniProtKB-AC was found
        ref_no_uniprot = df.loc[(df['RefSeq'].notna()) & (df['UniProtKB_AC'].isna()) ,'target'].to_list()
        if len(ref_no_uniprot) > 0:
            message = '{} RefSeq ids not mapped to UniProtKB-AC but included in NCBI-Code.'.format(len(ref_no_uniprot))
            self.console.print_warning(message)
            for id in ref_no_uniprot:
                logger.warning(f'Target sequence {id}') 

        # those where no UniProtKB-AC was found
        no_uniprot = df.loc[df['UniProtKB_AC'].isna(), 'target'].to_list()
        no_uniprot = list(set(no_uniprot).difference(set(ref_no_uniprot)))
        if len(no_uniprot) > 0:
            message = '{} ids not mapped to UniProtKB_AC.'.format(len(no_uniprot))
            self.console.print_warning(message)
            for id in no_uniprot:
                logger.warning(f'Target sequence {id}')   

        # those where no RefSeq or EMBL-CDS was found:
        no_ncbi = df.loc[df['ncbi_code'].isna(), 'target'].to_list()
        no_ncbi = list(set(no_ncbi).difference(set(no_uniprot)))
        if len(no_ncbi) > 0:
            message = '{} ids not mapped to any NCBI-Code (RefSeq or EMBL-CDS).'.format(len(no_ncbi))
            self.console.print_warning(message)
            for id in no_ncbi:
                logger.warning(f'Target sequence {id}')                 