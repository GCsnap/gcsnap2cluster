import os
import pandas

from gcsnap.configuration import Configuration
from gcsnap.rich_console import RichConsole
from gcsnap.genomic_context import GenomicContext
from gcsnap.parallel_tools import ParallelTools

from gcsnap.utils import split_dict_chunks

class Taxonomy:
    """
    Methods and attributes to extract the taxonomy.
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
        mode (str): The mode of the taxonomy search.
        chunks (int): The number of chunks to split the syntenies.
        msg (str): The message to display during the search.
        gc (GenomicContext): The GenomicContext object containing all genomic context information.
        database_path (str): The path to the database.
        taxonomy (dict): The dictionary with the taxonomy assigned to the flanking genes.
        console (RichConsole): The RichConsole object to print messages.
    """

    def __init__(self, config: Configuration, gc: GenomicContext):
        """
        Initialize the Taxonomy object.

        Args:
            config (Configuration): The Configuration object containing the arguments.
            gc (GenomicContext): The GenomicContext object containing all genomic context information.
        """        
        # get necessary configuration arguments      
        self.config = config
        self.gc = gc             
        self.chunks = (config.arguments['n_nodes']['value'] * config.arguments['n_ranks_per_node']['value']) - 1
        self.database_path = os.path.join(config.arguments['data_path']['value'],'db') 

        if config.arguments['get_taxonomy']['value']:
            self.mode = 'taxonomy'
            self.msg = 'Find and map taxonomy'
        else:
            self.mode = 'as_input'
            self.msg = 'Create input taxonomy dictionary. No taxomomies are searched'

        self.console = RichConsole()

    def get_taxonomy(self) -> dict:
        """
        Getter for the taxonomy attribute.

        Returns:
            dict: The dictionary with the taxonomy assigned to the flanking genes.
        """        
        return self.taxonomy

    def run(self) -> None:
        """
        Run the assignment of taxonomy to the flanking genes:
            - Find taxonomies
            - Add taxonomy to the flanking genes.
        Uses parallel processing with the processpool_wrapper from utils.py.
        """        

        if self.mode == 'taxonomy':
            self.clean_taxonomy = self.find_taxonomies(self.gc.get_all_taxids())

        # here we parallellize over chunks, so as many chunks as 
        # there are cores
        parallel_args = split_dict_chunks(self.gc.get_syntenies(), self.chunks)

        with self.console.status(self.msg):
            dict_list = ParallelTools.parallel_wrapper(parallel_args, self.run_each)
            # combine results
            # as this is a heavily nested dictionary, we need some recursive functionality
            self.taxonomy = self.merge_all_dicts(dict_list)

    def run_each(self, arg: dict) -> dict:
        """
        Run the create of the taxonomy dictionary for each target gene.

        Args:
            arg (dict): The dictionary with the target information.

        Returns:
            dict: The dictionary with the hierarchy of the taxonomy as nested dictionaries.
        """        
        content_dict = arg

        taxonomy = {}
        for target in content_dict.keys():
            ncbi_code = content_dict[target]['assembly_id'][0]
            taxID = content_dict[target]['flanking_genes']['taxID']
            species = content_dict[target]['flanking_genes']['species']
  
            genus = 'na'
            if self.mode == 'taxonomy':
                if species.split()[0] == 'uncultured':
                    genus = species.split()[1]
                else:
                    genus = species.split()[0]

            # take what is available from entrez taxonomies requests
            # if mode is as_input, we will have 'na' for all values as 
            # self.clean_taxonomy is empty
            superkingdom = self.clean_taxonomy.get(taxID,{}).get('superkingdom','na')
            phylum = self.clean_taxonomy.get(taxID,{}).get('phylum','na')
            taxclass = self.clean_taxonomy.get(taxID,{}).get('class','na')
            order = self.clean_taxonomy.get(taxID,{}).get('order','na')

            if self.mode == 'taxonomy':
                # now all are either correct string or 'na'
                if phylum == 'na':
                    phylum = '{}_na'.format(superkingdom)
                if taxclass == 'na':
                    taxclass = '{}_na'.format(phylum)
                if order == 'na':
                    order = '{}_na'.format(taxclass)

            # create the nested dictionaries
            taxonomy.setdefault(superkingdom, {}).setdefault(phylum, {}).setdefault(taxclass, {}) \
                .setdefault(order, {}).setdefault(genus, {}).setdefault(
                    species, {'ncbi_codes': [], 'target_members': []}
                )            

            taxonomy[superkingdom][phylum][taxclass][order][genus][species]['target_members'].append(target)
            taxonomy[superkingdom][phylum][taxclass][order][genus][species]['ncbi_codes'].append(ncbi_code)            

        return taxonomy

    def find_taxonomies(self, taxids: list) -> dict:
        """
        Find taxonomies for all flanking genes in the file rankedlineage.dmp.
        Details about the file can be found here:https://ftp.ncbi.nih.gov/pub/taxonomy/new_taxdump/taxdump_readme.txt

        Args:
            taxids (list): The list of taxids to find taxonomies for.
        """        
        # read the content
        with open(os.path.join(self.database_path,'rankedlineage.dmp'), 'r', encoding='utf-8') as f:
            content = f.read()

        lines = content.splitlines()
        parts = [line.strip().split('|') for line in lines]
        entries = [part for part in parts if part[0].strip() in [str(taxid) for taxid in taxids]]

        # extact
        # tax_id                -- node id
        # tax_name              -- scientific name of the organism
        # species               -- name of a species (coincide with organism name for species-level nodes)
        # genus					-- genus name when available
        # family				-- family name when available
        # order					-- order name when available
        # class					-- class name when available
        # phylum				-- phylum name when available
        # kingdom				-- kingdom name when available
        # superkingdom		    -- superkingdom (domain) name when available

        taxonomy_dmp = {e[0].strip() : {
                        'tax_name'  : e[1].strip() if len(e[1].strip()) > 0 else None,
                        'species'   : e[2].strip() if len(e[2].strip()) > 0 else None,
                        'genus'     : e[3].strip() if len(e[3].strip()) > 0 else None,
                        'family'    : e[4].strip() if len(e[4].strip()) > 0 else None,
                        'order'     : e[5].strip() if len(e[5].strip()) > 0 else None,
                        'class'     : e[6].strip() if len(e[6].strip()) > 0 else None,
                        'phylum'    : e[7].strip() if len(e[7].strip()) > 0 else None,
                        'kingdom'   : e[8].strip() if len(e[8].strip()) > 0 else None,
                        'superkingdom' : e[9].strip() if len(e[9].strip()) > 0 else None
                    }
                for e in entries
            }

        # remove all entries that are None
        # we do this here to have keep entrez result as is (to see what was not there)
        clean_taxonomy = {
            target: {key: value for key, value in sub_dict.items() if value is not None}
            for target, sub_dict in taxonomy_dmp.items()
        }

        return clean_taxonomy

    def find_taxonomies_new(self, taxids: list) -> dict:
        """
        Find taxonomies for all flanking genes in the file rankedlineage.dmp.
        Details about the file can be found here:https://ftp.ncbi.nih.gov/pub/taxonomy/new_taxdump/taxdump_readme.txt

        Args:
            taxids (list): The list of taxids to find taxonomies for.
        """        
        
        # Define the path to the file
        file_path = os.path.join(self.database_path, 'rankedlineage.dmp')
        
        # Read the .dmp file into a pandas DataFrame
        # Assuming the delimiter is '|' and the file has no header
        df = pandas.read_csv(file_path, sep='|', header=None, encoding='utf-8', skipinitialspace=True, engine='python')
        
        # Optionally, drop the last column if it's filled with NaN values
        df = df.dropna(axis=1, how='all')
        
        # If necessary, rename columns for better readability (optional)
        df.columns = ['tax_id','tax_name','species','genus','family','order','class','phylum','kingdom','superkingdom']  
            
        # Fill all empty cells with None
        df = df.where(pandas.notnull(df), None)
        
        # Remove tabs and any leading/trailing whitespace from all cells
        df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
        
        # Convert the list of taxids to int for comparison (thez might be strings as read from JSON)
        taxids_int = [int(taxid) for taxid in taxids]

        # Filter the DataFrame to get the entries with taxids in the provided list
        filtered_entries = df[df['tax_id'].isin(taxids_int)]
        # extact
        # tax_id          -- node id
        # tax_name        -- scientific name of the organism
        # species         -- name of a species (coincide with organism name for species-level nodes)
        # genus		      -- genus name when available
        # family				-- family name when available
        # order			   -- order name when available
        # class			   -- class name when available
        # phylum				-- phylum name when available
        # kingdom			-- kingdom name when available
        # superkingdom		-- superkingdom (domain) name when available

        taxonomy_dmp = {
            str(row.iloc[0]): {
                'tax_name': row.iloc[1],
                'species': row.iloc[2],
                'genus': row.iloc[3],
                'family': row.iloc[4],
                'order': row.iloc[5],
                'class': row.iloc[6],
                'phylum': row.iloc[7],
                'kingdom': row.iloc[8],
                'superkingdom': row.iloc[9]
            }
            for index, row in filtered_entries.iterrows()
        }

        # remove all entries that are None
        # we do this here to have keep entrez result as is (to see what was not there)
        clean_taxonomy = {
            target: {key: value for key, value in sub_dict.items() if value is not None}
            for target, sub_dict in taxonomy_dmp.items()
        }

        return clean_taxonomy

    def merge_nested_dicts(self, dict1: dict, dict2: dict) -> dict: 
        """
        Recursive method to merge two nested taxonomy dictionaries.

        Args:
            dict1 (dict): Dictionary to merge into.
            dict2 (dict): Dictionary to merge from.

        Returns:
            dict: The merged dictionary.
        """        
        for key, value in dict2.items():
            if key in dict1:
                if isinstance(value, dict) and isinstance(dict1[key], dict):
                    self.merge_nested_dicts(dict1[key], value)
                elif key == "ncbi_codes" or key == "target_members":
                    if dict1[key] == value:
                        dict1[key] = list(set(dict1[key] + value))
                    else:
                        dict1[key].extend(value)
                        dict1[key] = list(set(dict1[key]))
                else:
                    dict1[key] = value
            else:
                dict1[key] = value
        return dict1

    def merge_all_dicts(self, dict_list: list[dict]) -> dict:
        """
        Merge all nested dictionaries in a list.

        Args:
            dict_list (list[dict]): The list of dictionaries to merge.

        Returns:
            dict: The merged dictionary.
        """        
        merged_dict = dict_list[0]
        for d in dict_list[1:]:
            merged_dict = self.merge_nested_dicts(merged_dict, d)
        return merged_dict

