import json
import os

from gcsnap.mapping import SequenceMapping
from gcsnap.configuration import Configuration
from gcsnap.rich_console import RichConsole
from gcsnap.genomic_context import GenomicContext
from gcsnap.parallel_tools import ParallelTools

from gcsnap.utils import split_dict_chunks

class FamiliesFunctionsStructures:
    """ 
    Methods and attributes to get the functional annotations and structures of the families.
    Here we do not have data locally. Hence, the annotations are taken from a specified File.
   
    Out idea is to create those via the EBI API, but this is not executed yet.
    Each annotation file contains 1 uniprot annotations and is named with the uniprot accession!
    The code to get all the files is at the bottom based on the EBI API from GCsnap2.0 Desktop
    Once all files or many are downloade, we use those files to get the functional annotations.

    For the 3D strucutre, we just combine the SWISS-MODEL URL with the UniProt Code.
    There is no verification, that the model exists!!
    
    Attributes:
        config (Configuration): The Configuration object containing the arguments.
        get_pdb (bool): The flag to get the PDB structures.
        annotation_files_path (str): The path to the functional annotation files. If None, no annotations are retrieved.
        families (dict): The dictionary with the families and their members.
        console (RichConsole): The RichConsole object to print messages.
        annotations_and_structures (dict): The dictionary with the functional annotations and structures
    """

    def __init__(self, config: Configuration, gc: GenomicContext):
        """
        Initialize the FamiliesFunctionsStructures object.

        Args:
            config (Configuration): The Configuration object containing the arguments.
            gc (GenomicContext): The GenomicContext object containing all genomic context information.
        """        
        self.config = config
        #self.chunks = (config.arguments['n_nodes']['value'] * config.arguments['n_ranks_per_node']['value']) - 1
        self.get_pdb = config.arguments['get_pdb']['value']
        self.annotation_files_path = config.arguments['functional_annotation_files_path']['value']        
        self.annotations_and_structures = {}

        if self.annotation_files_path is not None:
            self.get_annotation = True
        else:
            self.get_annotation = False

        # set parameters
        self.families = gc.get_families()

        self.console = RichConsole()

    def get_annotations_and_structures(self) -> dict:
        """
        Getter for the annotations_and_structures attribute.

        Returns:
            dict: The dictionary with the functional annotations and structures.
        """        
        return self.annotations_and_structures

    def run(self) -> None:
        """
        Run the retrieval of the functional annotations and structures:
            - Map every family member to UniProtKB-AC.
            - Request all annotation information from EbiAPI in parallel.
            - Extract the functional annotations and structures for each family in parallel.
        Uses parallel processing with the parallel_wrapper from ParallelTools. 
        Only done when conserved families were found.
        """        
        if (list(self.families.keys()) == [0] or list(self.families.keys()) == [1] or
                    list(self.families.keys()) == [-1, 0]):
            self.console.print_warning('No conserved family found and no functional annotation possible')
        elif self.get_pdb or self.get_annotation:
            self.console.print_step('Functional annotation needs mapping first')
            # find all family members
            # always exclude non-conserved families and pseudogenes as uniprot mapping
            all_members = [member_code for family in self.families.keys() 
                           for member_code in self.families[family]['members']
                           if 0 < family]
            
            # map all members to UniProtKB-AC
            mapping = SequenceMapping(self.config, all_members, 'for functional annotation')
            mapping.run()
            mapping_dict = mapping.get_target_to_result_dict('UniProtKB_AC')

            # create parallel args with 4 items
            parallel_args = [({k: v}, mapping_dict, self.get_pdb, self.get_annotation)
                             for k,v in self.families.items()]

            with self.console.status('Get functional annotations and structures'):
                result_list = ParallelTools.parallel_wrapper(parallel_args, self.run_each)
                # combine results
                self.annotations_and_structures = {k: v for dict_ in result_list for k, v in dict_.items()}
        else:
            self.console.print_skipped_step('No annotations or structures retrieval requested')

    def run_each(self, args: tuple[dict,dict,bool,bool]) -> dict:
        """
        Run the retrieval of the functional annotations and structures for each family
        used in parallel processing.

        Args:
            args (tuple[dict,dict,dict,bool,bool]): The arguments for the retrieval.
                First element is a dictionary with the family and its members.
                Second element is a dictionary with the mapping of the members to UniProtKB-AC.
                Third element is a flag to get the PDB structures.
                Fourth element is a flag to get the functional annotations.

        Returns:
            dict: The dictionary with the family and its members with the functional annotations and structures.
        """        
        families, mapping_dict, get_pdb, get_annotations = args

        # all families that are not pseudogenes
        # a list of tuples (family, member)
        family_keys = [(key, sorted(families[key]['members'])) 
                        for key in sorted(list(families.keys()))
                        if 'function' not in families[key]]
        
        family_function = {}
        family_structure = ''
        family_uniprot_code = ''
        family_model_state = 'Model does not exist'

        for family, members in family_keys:

            # pseudo genes and non-conserved families
            if family == 0 or family == -1:
                family_function = {}
                family_structure = 'n.a.'
                family_uniprot_code = 'n.a.'
                family_model_state = 'n.a.'

            # conserved families
            else:
                family_function = {}
                family_structure = ''
                family_uniprot_code = ''
                family_model_state = 'Model does not exist'

                # it is done until a result is found
                for member in members:
                    # if not found, it will be nan
                    uniprot_code = mapping_dict.get(member,'nan')

                    if (family_uniprot_code == '' or family_structure == '' or 
                        family_function == {} or family_function['Function_description'] == ''):
                    
                        if get_pdb and family_structure == '' and uniprot_code != 'nan':
                            
                            # this part is the first problem without API access
                            # it is not possible to check if the model exists
                            # we just combine the URL with the UniProt code if that exists

                            # check either in SwissModel or AlphaFold
                            curr_pdb = 'https://swissmodel.expasy.org/repository/uniprot/{}'.format(uniprot_code)
                            if 'nan' in curr_pdb:
                                curr_pdb = 'https://alphafold.ebi.ac.uk/entry/{}'.format(uniprot_code)

                            if 'nan' not in curr_pdb:
                                family_structure = curr_pdb
                                family_uniprot_code = uniprot_code
                                family_model_state = 'Model exists'
                            elif uniprot_code != 'nan' and curr_pdb == 'nan*':
                                family_uniprot_code = uniprot_code

                        if (get_annotations and (family_function == {} 
                                                 or family_function['Function_description'] == '')):        
                            
                            # get functional annotations
                            uniprot_accession = uniprot_code.split('_')[0]
                            uniprot_data = self.load_annotation_file(os.path.join(self.annotation_files_path, 
                                                                                '{}.json'.format(uniprot_accession)))
                            curr_uniprot_annotations = self.parse_uniprot_data(uniprot_data, 
                                                                    previous_annotations = family_function)
                            if curr_uniprot_annotations != 'nan':
                                family_function = curr_uniprot_annotations

                            if uniprot_code != 'nan' and family_structure == '' and family_uniprot_code == '':
                                family_uniprot_code = uniprot_code

                if family_uniprot_code == '':
                    family_model_state = 'Not possible to map'

            families[family]['function'] = 'n.a.' if family_function == {} else family_function
            families[family]['structure'] = family_structure
            families[family]['uniprot_code'] = family_uniprot_code
            families[family]['model_state'] = family_model_state

        return families

    def load_annotation_file(self, file_name: str) -> dict:
        """
        Load the annotation file containing the functional annotations and structures.

        Returns:
            dict: The dictionary with the functional annotations and structures.
        """        
        file = os.path.join(self.annotation_files_path, file_name) 
        try:
            with open(os.path.join(file), 'r') as file:
                return json.load(file)
        except FileNotFoundError:
            self.console.print_warning('Your specifed file {} could not be found.'.format(file)) 
            return {}
        
    def parse_uniprot_data(self, uniprot_data: dict, previous_annotations: dict = {}) -> dict:
        """
        Parse the data of a protein from the EBI database and extract annotations.

        Args:
            uniprot_data (dict): The data of the protein.
            previous_annotations (dict, optional): Previously found annotations of the protein. Defaults to {}.

        Returns:
            dict: The annotations of the protein.
        """        
        if previous_annotations == {}:
            uniprot_annotations = {'TM_topology': '', 'GO_terms': [], 'Keywords': [],
                                   'Function_description': ''}
        else:
            uniprot_annotations = previous_annotations

        if 'features' in uniprot_data:
            for feature in uniprot_data['features']:
                if feature['type'] == 'TRANSMEM' and uniprot_annotations['TM_topology'] == '':
                    tm_topology = feature['description']

                    if 'Helical' in tm_topology:
                        tm_topology = 'alpha-helical'
                    elif 'Beta' in tm_topology:
                        tm_topology = 'beta-stranded'
                    else:
                        tm_topology = 'transmembrane'

                    uniprot_annotations['TM_topology'] = tm_topology

        if 'dbReferences' in uniprot_data:
            for dbReference in uniprot_data['dbReferences']:
                if dbReference['type'] == 'GO':
                    go_term = dbReference['properties']['term']
                    if go_term not in uniprot_annotations['GO_terms']:
                        uniprot_annotations['GO_terms'].append(go_term)

        if 'keywords' in uniprot_data:
            for keyword in uniprot_data['keywords']:
                keyword_term = keyword['value']
                if (keyword_term not in uniprot_annotations['Keywords'] 
                    and keyword_term != 'Reference proteome'):
                    uniprot_annotations['Keywords'].append(keyword_term)

        if 'comments' in uniprot_data:
            for comment in uniprot_data['comments']:
                if comment['type'] == 'FUNCTION' and uniprot_annotations['Function_description'] == '':
                    uniprot_annotations['Function_description'] = comment['text'][0]['value']

        return uniprot_annotations            