import os
import gzip # to work with .gz

from gcsnap.configuration import Configuration
from gcsnap.rich_console import RichConsole
from gcsnap.genomic_context import GenomicContext
from gcsnap.db_handler_assemblies import AssembliesDBHandler
from gcsnap.parallel_tools import ParallelTools

from gcsnap.utils import split_list_chunks
from gcsnap.utils import WarningToLog

import logging
logger = logging.getLogger('iteration')

class Assemblies:    
    """
    Methods and attributes to query assembly accession and infos from database
    and parse flanking genes.
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
        n_flanking5 (int): Number of flanking genes to extract at the 5' end of target.
        n_flanking3 (int): Number of flanking genes to extract at the 3' end of target.
        chunks (int): Number of chunks to split the syntenies.
        exclude_partial (bool): Exclude partial genomic blocks.
        database_path (str): Path to directory containing the database.
        data_path (str): Path to the parent directory with the assembly files folders for genbank and refseq.
        config (Configuration): Configuration object.
        console (RichConsole): Console object.
        targets_and_ncbi_codes (list): List of tuples with target and ncbi code.
    """

    def __init__(self, config: Configuration, mappings: list[tuple[str,str]]) -> None:                     
        """
        Initialize the Assemblies object.

        Args:
            config (Configuration): Configuration object containing the arguments.
            mappings (list[tuple[str,str]]): Contains the target and its ncbi code.
        """        
        # get necessary configuration arguments        
        self.n_flanking5 = config.arguments['n_flanking5']['value']  
        self.n_flanking3 = config.arguments['n_flanking3']['value']
        self.chunks = (config.arguments['n_nodes']['value'] * config.arguments['n_ranks_per_node']['value'])        
        self.exclude_partial = config.arguments['exclude_partial']['value']
        self.database_path = os.path.join(config.arguments['data_path']['value'],'db') 
        self.data_path = os.path.join(config.arguments['data_path']['value']) 
        self.config = config

        self.console = RichConsole()

        # input list with [(target, ncbi_code)]
        self.targets_and_ncbi_codes = mappings

    def get_flanking_genes(self) -> dict:
        """
        Getter for the flanking_genes attribute.

        Returns:
            dict: The flanking gene infrmation.
        """        
        return self.flanking_genes
    
    def run(self) -> None:
        """
        Run the process to query information from DB and extract flanking genes for the targets:
            - Split into chunks and start parallel execution.
        Uses parallel processing with the parallel_wrapper from ParallelTools. 
        """        
        # split input into chunks
        # we don't want to have only as many chunks as processes to have better load
        # balance as the assembly files are very different in size
        # this is acutally set arbitrarily
        parallel_args = split_list_chunks(self.targets_and_ncbi_codes, self.chunks)

        with self.console.status('Download assemblies and extract flanking genes'):
            dict_list = ParallelTools.parallel_wrapper(parallel_args, self.run_each)
            # # uncomment this to exctract a list of needed assembly files to run evaluation. Those were added to the sequences.db
            # return dict_list
        
            # combine results
            self.flanking_genes = {k: v for d in dict_list for k, v in d.items() 
                                   if v.get('flanking_genes') is not None}
            not_found = {k: v for d in dict_list for k, v in d.items() 
                         if v.get('flanking_genes') is None and v.get('msg').startswith('File')}
            partial = {k: v for d in dict_list for k, v in d.items()
                          if v.get('flanking_genes') is None and v.get('msg').startswith('Partial')}
            
        if not_found:
            self.log_not_found(not_found)
        if partial:
            self.log_partial(partial)

        if len(self.targets_and_ncbi_codes) != len(self.flanking_genes):
            # check here for those no information was in the found
            not_found = [target[0] for target in self.targets_and_ncbi_codes 
                         if target[0] not in self.flanking_genes 
                         and target[0] not in not_found
                         and target[0] not in partial]
            missing = {target: {'msg': 'No assembly info found in databse'} for target in not_found}
            self.log_not_found(missing)

        if not self.flanking_genes: 
            msg = 'No flanking genes found for any target sequence. Continuing is not possible.'
            self.console.stop_execution(msg = msg)

    def run_each(self, args: list[tuple[str,str]]) -> dict[str, dict]:
        """
        Called in parallel and is doing:
            - Get assembly accessions from database.
            - Get assembly urls, species and tax ID from database.
            - Extract flanking genes from files.           

        Args:
            args (list[tuple[str,str]]): List of tuples with the target and its ncbi code.

        Returns:
            dict[str, dict]: The flanking genes information for the targets.
        """        
        target_tuples = args
        # get the assembly accessions
        accession_tuples = self.get_assembly_accessions(target_tuples)

        # get the assembly urls
        # the result is a list of tuples with (target, ncbi_code, accession, url, taxid, species)
        info_tuples = self.get_assembly_info(accession_tuples)

        # # uncomment this to exctract a list of needed assembly files to run evaluation. Those were added to the sequences.db
        # return info_tuples 

        # loop over all targets and download and extract flanking genes
        flanking_genes = {}
        for element in info_tuples:
            # merge them to results
            flanking_genes |= self.read_and_parse_assembly(element)

        return flanking_genes

    def read_and_parse_assembly(self, element: tuple[str,str,str,str,str,str]) -> dict:
        """
        Wrapper to read and parse the assembly file.

        Args:
            element (tuple[str,str,str,str,str,str]): Tuple with target, ncbi code, accession, url, taxid, species. 

        Returns:
            dict: The flanking genes information for the target.
        """        
        target, ncbi_code, accession, url, taxid, species = element
        
        try:
            assembly_path = self.get_gz_path(url)
            lines = self.read_gz_file(assembly_path)
            flanking_genes = self.parse_assembly(ncbi_code, lines)

            # add species and taxid to flanking genes
            flanking_genes['species'] = species
            flanking_genes['taxID'] = taxid
            # add spicies again to syntenies
            return {target: {'flanking_genes': flanking_genes,
                             'assembly_id':  [ncbi_code, accession, url],
                             'species': species}}   
        except WarningToLog as e:
            # return None for flanking genes and message, logged later
            return {target: {'flanking_genes': None,
                             'msg': str(e)}}
    
    def get_assembly_accessions(self, target_tuples : list[tuple[str,str]]) -> list[tuple[str,str,str]]:
        """
        Query the assembly accession for the ncbi codes from the database.
        Those are added to each input tuples.

        Args:
            target_tuples (list[tuple[str,str]]): List of tuples with the target and its ncbi code.

        Returns:
            list[tuple[str,str,str]]: List of tuples with the target, ncbi code and assembly accession.
        """        
        # all ncbi_codes
        ncbi_codes = [element[1] for element in target_tuples]

        # select from database
        assembly_db = AssembliesDBHandler(os.path.join(self.database_path))
        # get the assemblies accession for the ncbi codes (from default table 'mapping')
        result_tuples = assembly_db.select(ncbi_codes, request_size = 5000)

        # combine the targets, the ncbi_codes and the acessions
        return [(target, ncbi, accession) for target, ncbi in target_tuples 
                            for result_ncbi, accession in result_tuples if ncbi == result_ncbi]
    
    def get_assembly_info(self, target_tuples: list[tuple[str,str,str]]) -> list[tuple[str,str,str,str,str,str]]:
        """
        Query the assembly urls, taxid and species for the assembly accessions from the database.
        Those are added to each input tuples.

        Args:
            target_tuples (list[tuple[str,str,str]]): List of tuples with the target, ncbi code and assembly accession.

        Returns:
            list[tuple[str,str,str,str,str,str]]: List of tuples with the target, ncbi code, assembly accession, url, taxid and species.
        """        
        assembly_accessions = [element[2] for element in target_tuples]

        assembly_db = AssembliesDBHandler(os.path.join(self.database_path))
        # get the url and taxid and the species name for the assembly accessions
        result_tuples = assembly_db.select(assembly_accessions, table = 'assemblies', request_size = 20000)

        # combine the targets, the ncbi_codes and the acessions
        return [(target, ncbi, accession, url, taxid, species) 
                            for target, ncbi, accession in target_tuples 
                            for result_accession, url, taxid, species in result_tuples 
                            if accession == result_accession]
    
    def get_gz_path(self, url: str) -> str:
        """
        Determin the path to the assembly files.

        Args:
            url (str): The url of the assembly file as stored in the database.

        Returns:
            str: The path to the assembly file on the cluster.
        """        
        ass_file = os.path.basename(url) + '_genomic.gff.gz'

        # search the file
        if ass_file.startswith('GCA'):
            db = 'genbank'
        else:
            db = 'refseq'
        return os.path.join(self.data_path, db, 'data', ass_file)
   
    def read_gz_file(self, file_path: str) -> list:
        """
        Read the content of a .gz file.

        Args:
            file_path (str): The path of the .gz file.

        Returns:
            list: The content of the file as a list of lines.
        """      
        try:  
            with gzip.open(file_path, 'rt', encoding='utf-8') as file:
                content = file.read()
            return content.splitlines()   
        except FileNotFoundError:
            raise WarningToLog('File {} not found'.format(file_path))             

    def parse_assembly(self, ncbi_code: str, lines: list) -> dict:
        """
        Wrapper to extract the genomic context block and to extract the flanking genes.

        Args:
            ncbi_code (str): The NCBI code.
            lines (list): The content of the assembly file.

        Returns:
            dict: The flanking genes.
        """        
        genomic_context_block = self.extract_genomic_context_block(ncbi_code, lines)
        return self.parse_genomic_context_block(ncbi_code, genomic_context_block)

    def extract_genomic_context_block(self, target_ncbi_code: str, lines: list) -> list:
        """
        Extract first all lines belonging to the scaffold containing the target gene.
        Then extract the genomic context block (n_flanking5 on 5' end and 
        n_flanking3 on 3' end based on the direction of the target) from that scaffold.
        One line of the genomic context block looks like this:
            JQ926483.1	Genbank	CDS	1	668	.	+	0
            ID=cds-AFI40896.1;Parent=gene-VP1;Dbxref=NCBI_GP:AFI40896.1;
            Name=AFI40896.1;end_range=668,.;gbkey=CDS;gene=VP1;partial=true;
            product=RNA-dependent RNA polymerase;protein_id=AFI40896.1;start_range=.,1

        Args:
            target_ncbi_code (str): The NCBI code of the target gene.
            lines (list): The content of the assembly file.

        Returns:
            list: The lines containing the flanking genes.
        """        
         # line numbers where different scaffolds (sequence regions) start
        scaffold_positions = [0] + [index for index, val in enumerate(lines) if 
                            val.startswith('##sequence-region')] + [len(lines)-1]   
             
        # line number of target
        target_position = [index for index, val in enumerate(lines) 
                    if 'ID=cds-{}'.format(target_ncbi_code) in val or
                    'Name={}'.format(target_ncbi_code) in val or
                    'protein_id={}'.format(target_ncbi_code) in val]
        
        if not target_position:
            raise WarningToLog('{} not found in'.format(target_ncbi_code))

        # select scaffold region with target
        region_start_end = [(scaffold_positions[i], scaffold_positions[i + 1]) for i in 
                range(len(scaffold_positions) - 1) if scaffold_positions[i] 
                <= target_position[0] <= scaffold_positions[i + 1]]
        
        # extract the scaffold and lines containing coding sequence (CDS) information
        scaffold = [line for line in lines[region_start_end[0][0]:region_start_end[0][1]] if 
                    (len(line.split('\t')) >= 3 and line.split('\t')[2] == 'CDS')]
        
        # target position in scaffold
        index_of_target = [index for index, val in enumerate(scaffold) 
                           if 'ID=cds-{}'.format(target_ncbi_code) in val or
                           'Name={}'.format(target_ncbi_code) in val or
                           'protein_id={}'.format(target_ncbi_code) in val][0]
                
        # need to know direction to define what flanking genes to extract
        direction_of_target = scaffold[index_of_target].split('\t')[6]
        
        # define neighbor indizes depending on direction
        if direction_of_target == '+':
            # upper slice index can be out of bounds without error, lower must be non negative
            start = 0 if (index_of_target - self.n_flanking5) < 0 else index_of_target - self.n_flanking5
            end = index_of_target + self.n_flanking3 + 1
            # extract the genomic context 
            genomic_context_block = scaffold[start:end]
        else:
            # upper slice index can be out of bounds without error, lower must be non negative
            start = 0 if (index_of_target - self.n_flanking3) < 0 else index_of_target - self.n_flanking3
            end = index_of_target + self.n_flanking5 + 1
            # extract the genomic context 
            genomic_context_block = scaffold[start:end]
            # reverse if direction of target is '-'
            genomic_context_block = genomic_context_block[::-1]
        
        # exclude partials if desired
        if self.exclude_partial and len(genomic_context_block) < (self.n_flanking5 + self.n_flanking3 + 1):
            raise WarningToLog('Partial genomic block for {} excluded!'.format(target_ncbi_code))
        
        return genomic_context_block

    def parse_genomic_context_block(self, target_ncbi_code: str, genomic_context_block: list) -> dict:
        """
        Parse the genmoic context block to extract the flanking genes information.
        One line of the genomic context block looks like this:
            JQ926483.1	Genbank	CDS	1	668	.	+	0
            ID=cds-AFI40896.1;Parent=gene-VP1;Dbxref=NCBI_GP:AFI40896.1;
            Name=AFI40896.1;end_range=668,.;gbkey=CDS;gene=VP1;partial=true;
            product=RNA-dependent RNA polymerase;protein_id=AFI40896.1;start_range=.,1

        Args:
            target_ncbi_code (str): The NCBI code of the target gene.
            genomic_context_block (list): The lines containing the flanking genes.

        Returns:
            dict: The extracted flanking genes information.
        """
        # result dictionary
        flanking_genes = GenomicContext.get_empty_flanking_genes()

        # parse the genomic context
        for line in genomic_context_block:   

            line_data = line.split('\t')            
            start = int(line_data[3])
            end = int(line_data[4])
            direction = line_data[6]
            
            if 'cds-' in line:
                ncbi_code = line_data[8].split('ID=cds-')[1].split(';')[0]
            elif 'Name=' in line:
                ncbi_code = line_data[8].split('Name=')[1].split(';')[0]
            else:
                ncbi_code = 'unk'

            if 'pseudo=' not in line and 'product=' in line and 'fragment' not in line:
                prot_name = line_data[8].split('product=')[1].split(';')[0]
            else:
                prot_name = 'pseudogene'

            # it means that this is some kind of fragmented gene (has introns?...) 
            # and so we have to collect the largest interval encompassing it
            if ncbi_code in flanking_genes['ncbi_codes'] and flanking_genes['ncbi_codes'][-1] == ncbi_code: 
                if start < flanking_genes['starts'][-1]:
                    flanking_genes['starts'][-1] = start
                if end > flanking_genes['ends'][-1]:
                    flanking_genes['ends'][-1] = end
            else:
                if '|' in ncbi_code:
                    ncbi_code = ncbi_code.replace('|','_')

                flanking_genes['ncbi_codes'].append(ncbi_code)
                flanking_genes['names'].append(prot_name)
                flanking_genes['starts'].append(start)
                flanking_genes['ends'].append(end)
                flanking_genes['directions'].append(direction)
        
        # index of target in flanking genes
        index_of_target = flanking_genes['ncbi_codes'].index(target_ncbi_code)

        # direction of target
        direction_of_target = flanking_genes['directions'][index_of_target]
        
        # add relative starts and ends depending on direction
        if direction_of_target == '+':
            for key in ['starts','ends']:
                lst = [e - flanking_genes['starts'][index_of_target] + 1 for e in flanking_genes[key]]
                flanking_genes['relative_{}'.format(key)] = lst  
        else:
            # order is reversed, old ends determin the starts and vice-versa
            # old end of target is the new base point
            lst = [flanking_genes['ends'][index_of_target] - e + 1 for e in flanking_genes['ends']]
            flanking_genes['relative_starts'] = lst  
            lst = [flanking_genes['ends'][index_of_target] - e + 1 for e in flanking_genes['starts']]
            flanking_genes['relative_ends'] = lst  
            
            # turn + and - directions
            flanking_genes['directions'] = ['+' if d == '-' else '-' for d in flanking_genes['directions']]
            
        return flanking_genes       

    def log_not_found(self, not_found: dict) -> None:
        """
        Write the targets for which no flanking genes were found to the log file.

        Args:
            not_found (dict): The targets for which no flanking genes were found.
        """        
        message = 'No flanking genes found for {} target sequences.'.format(len(not_found))
        self.console.print_warning(message)
        for k,v in not_found.items():
            logger.warning('For target {}: {}'.format(k,v.get('msg'))) 

    def log_partial(self, partial: dict) -> None:
        """
        Write the targets for which partial genomic context blocks were found to the log file.

        Args:
            not_found (dict): The targets for which partial blocks were found.
        """        
        message = 'Partial genomic blocks for {} target sequences excluded. Set --exclude-partial False to include.'.format(len(partial))
        self.console.print_warning(message)
        for k,v in partial.items():
            logger.warning('For target {}: {}'.format(k,v.get('msg')))             
