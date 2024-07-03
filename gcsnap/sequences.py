
import json

from gcsnap.rich_console import RichConsole
from gcsnap.configuration import Configuration
from gcsnap.entrez_query import EntrezQuery

from gcsnap.utils import processpool_wrapper

class Sequences:
    def __init__(self, config: Configuration, flanking_genes: dict[str, dict]):
        self.config = config
        # get necessary configuration arguments        
        self.cores = config.arguments['n_cpu']['value'] 

        # set arguments
        self.flanking_genes = flanking_genes

        self.console = RichConsole()

    def run(self) -> None:
        # Extract all ncbi_codes lists from flanking_genes
        all_ncbi_codes = [code for target in self.flanking_genes.values() 
                          for code in target['flanking_genes'].get('ncbi_codes', [])]
        self.find_sequences(all_ncbi_codes)

        # Prepare a list of tuples (target, dict_for_target)
        parallel_args = [(target, target_dict) for target, target_dict in self.flanking_genes.items()]

        with self.console.status('Add sequences, tax id and species name to flanking genes'):
            dict_list = processpool_wrapper(self.cores, parallel_args, self.run_each)
            # combine results
            self.genomic_context = {k: v for d in dict_list for k, v in d.items()}

        # dump information to file
        self.write_syntenies()

    def run_each(self, args: tuple[str,dict]) -> dict:
        target, content_dict = args
        # update flanking genes with sequence
        sequences = [self.get_sequence(ncbi_code) for ncbi_code in content_dict['flanking_genes']['ncbi_codes']]
        content_dict['flanking_genes']['sequences'] = sequences

        # add species and taxid for target_ncbi code (first one in the list)
        target_ncbi = content_dict['assembly_id'][0]
        # species in contained twice in the dict
        content_dict['flanking_genes']['species'] = self.get_species(target_ncbi)
        content_dict['species'] = content_dict['flanking_genes']['species']
        content_dict['flanking_genes']['taxID'] = self.get_taxid(target_ncbi)

        return {target: content_dict}

    def find_sequences(self, ncbi_codes: list) -> None:
        # get the information for all ncbi codes
        # Noteworthy: While for accessions there were at most as many series as targets
        # here it is done for all flanking genes (1 + n_flanking_3 + n_flanking_5)
        entrez = EntrezQuery(self.config, ncbi_codes, db='protein', rettype='fasta', 
                             retmode='xml', logging=True)
        self.sequences = entrez.run()

    def get_sequence(self, ncbi_code: str) -> str:
        entry = self.sequences.get(ncbi_code, {})        
        return entry.get('seq', 'FAKESEQUENCEFAKESEQUENCEFAKESEQUENCEFAKESEQUENCE')

    def get_taxid(self, ncbi_code: str) -> str:
        entry = self.sequences.get(ncbi_code, {})        
        return entry.get('taxid','')
    
    def get_species(self, ncbi_code: str) -> str:
        entry = self.sequences.get(ncbi_code, {})        
        return entry.get('species','')    
    
    def get_genomic_context(self) -> dict:
        return self.genomic_context
    
    def write_syntenies(self) -> None:
        with open('genomic_context_information.json', 'w') as file:
            json.dump(self.genomic_context, file, indent = 4)   
        self.config.print_done('Genomic context information written to genomic_context_information.json')     