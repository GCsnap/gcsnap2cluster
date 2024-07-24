import os
import subprocess
import numpy as np
# pip install scipy
from scipy.cluster import hierarchy
from scipy.spatial import distance

from gcsnap.configuration import Configuration
from gcsnap.genomic_context import GenomicContext
from gcsnap.rich_console import RichConsole

class MMseqsCluster:
    """ 
    Methods and attributes to cluster flanking genes using MMseqs2.
    Needs MMseqs2 installed or the path to the executable set either in the config.yaml 
    or as a CLI argument.

    Attributes:
        config (Configuration): The Configuration object containing the arguments.
        cores (int): The number of CPU cores to use.
        max_evalue (float): The maximum e-value for the search.
        min_coverage (float): The minimum coverage for the search.
        num_iterations (int): The number of iterations for the search.
        mmseqs_executable (str): The path to the MMseqs2 executable.
        default_base (int): The default base for the distance matrix.
        gc (GenomicContext): The GenomicContext object containing all genomic context information.
        out_dir (str): The path to store the output of MMseqs.
        sensitivity (float): The sensitivity for the search.
        console (RichConsole): The RichConsole object to print messages.
    """

    def __init__(self, config: Configuration, gc: GenomicContext, out_dir: str):
        """
        Initialize the MMseqsCluster object

        Args:
            config (Configuration): The Configuration object containing the arguments.
            gc (GenomicContext): The GenomicContext object containing all genomic context information.
            out_dir (str): The path to store the output.
        """        
        self.config = config
        self.cores = config.arguments['n_cpu']['value']
        self.max_evalue = config.arguments['max_evalue']['value']
        self.min_coverage = config.arguments['min_coverage']['value']
        self.num_iterations = config.arguments['num_iterations']['value']
        self.mmseqs_executable = r'{}'.format(config.arguments['mmseqs_executable_path']['value'])
        self.default_base = config.arguments['default_base']['value']

        # set arguments
        self.gc = gc
        # if mmseqs temporary folder is not set, use the output folder
        mmseqs_out_dir = config.arguments['tmp_mmseqs_folder']['value']
        if mmseqs_out_dir is None:
            self.out_dir = out_dir
        else:
            self.out_dir = mmseqs_out_dir
        # check if existing
        if not os.path.isdir(self.out_dir):
            os.mkdir(self.out_dir)            
  
        self.sensitivity = 7.5

        self.console = RichConsole()

    def run(self) -> None:
        """
        Run the clustering of flanking genes using MMseqs2:
            - Prepare data for MMseqs
            - Run MMseqs
            - Extract distance matrix
            - Find clusters
            - Mask singleton clusters
        """        
        with self.console.status('Prepare data for MMseqs'):
            self.fasta_file = self.gc.write_to_fasta('flanking_sequences.fasta', 
                                            self.out_dir, exclude_pseudogenes = False)  
            self.cluster_order = self.gc.get_fasta_order(exclude_pseudogenes = False) 
        with self.console.status('Running MMseqs'):            
            self.run_mmseqs()
        with self.console.status('Extracting distance matrix'):
            self.extract_distance_matrix()
        with self.console.status('Find clusters'):
            self.find_clusters()
            self.mask_singleton_clusters()        

    def get_distance_matrix(self) -> np.array:
        """
        Getter for the distance_matrix attribute.

        Returns:
            np.array: The distance matrix.
        """        
        return self.distance_matrix

    def get_clusters_list(self) -> list[int]:
        """
        Getter for the cluster_list attribute.

        Returns:
            list[int]: The list of clusters.
        """        
        return self.cluster_list      

    def get_cluster_order(self) -> list[str]:
        """
        Getter for the cluster_order attribute.

        Returns:
            list[str]: The order of the clusters.
        """        
        return self.cluster_order       

    def run_mmseqs(self) -> None:
        """
        Run MMseqs to cluster flanking genes.

        Raises:
            FileNotFoundError: If MMseqs is not installed or the path to executable is wrongly set.
        """            
        self.mmseqs_results = os.path.join(self.out_dir, '{}_{}.mmseqs'.format(
            os.path.basename(self.fasta_file)[:-6], self.max_evalue))
        
        if not os.path.isfile(self.mmseqs_results):
            try:
                _, stderr = self.mmseqs_command('mmseqs')
                if len(stderr) > 0:
                    raise FileNotFoundError
            except FileNotFoundError:
                try:
                    # do again with the specified executable
                    _, stderr = self.mmseqs_command(self.mmseqs_executable)
                    if not os.path.isfile(self.mmseqs_results):
                        raise NoResultsError
                except NoResultsError:
                    self.console.print_error('MMseqs did not run properly') 
                    self.console.print_hint('Try to clean the temporary folder (e.g., rm -rf /tmp/* on Linux) and run again.') 
                    exit(1)
                except:
                    self.console.print_error('No MMseqs installation was found') 
                    self.console.print_hint('Please install MMseqs or add the path to the executable to config.yaml.')
                    exit(1)                    

    def mmseqs_command(self, mmseqs: str) -> tuple:
        """
        Run MMseqs command to execute.

        Args:
            mmseqs (str): Either 'mmseqs' or if not installed, the path to the MMseqs executable.

        Returns:
            tuple: The stdout and stderr of the MMseqs command.
        """        
        # returns stdout,stderr
        command = [mmseqs, 
                'easy-search', 
                self.fasta_file, 
                self.fasta_file, 
                self.mmseqs_results, 
                self.out_dir, 
                '-e', str(self.max_evalue), 
                '-s', str(self.sensitivity),
                '-c', str(self.min_coverage),
                '--num-iterations', str(self.num_iterations),
                '--threads', str(self.cores),
                '--format-output',
                'query,target,evalue']
        
        result = subprocess.run(command, capture_output=True, text=True, shell=True)        
        return result.stdout, result.stderr       
    
    def extract_distance_matrix(self) -> None:
        """
        Extract the distance matrix from the MMseqs results.
        """        
        # crate base distance matrix
        distance_matrix = [[self.default_base if i!=j else 0 for i in self.cluster_order] 
                        for j in self.cluster_order]
        queries_labels = {query: i for i, query in enumerate(self.cluster_order)}

        # read mmseqs results
        with open(self.mmseqs_results, 'r') as f:
            mmseqs_records = f.readlines()

        for hsp in mmseqs_records:
            hsp = hsp.split()
            if len(hsp) > 0:
                query = hsp[0].split('|')[0]
                query_index = queries_labels[query]
                target = hsp[1].split('|')[0]
                if target != query:
                    target_index = queries_labels[target]
                    distance_matrix[query_index][target_index] = 0
                    distance_matrix[target_index][query_index] = 0

        self.distance_matrix = np.array(distance_matrix)      

    def find_clusters(self, t: int = 0) -> None:
        """
        Find clusters using the distance matrix.

        Args:
            t (int, optional): The threshold for the clustering. Defaults to 0.
        """        
        distance_matrix = distance.squareform(self.distance_matrix)
        linkage = hierarchy.linkage(distance_matrix, method = 'single')
        clusters = hierarchy.fcluster(linkage, t, criterion = 'distance')
        self.cluster_list = [int(i) for i in clusters]

    def mask_singleton_clusters(self, mask: int = 0) -> None:
        """
        Mask singleton clusters.

        Args:
            mask (int, optional): The value to mask the singleton clusters. Defaults to 0.
        """        
        self.cluster_list = [mask if list(self.cluster_list).count(value) == 1 
                             else value for value in self.cluster_list]      
        
class NoResultsError(Exception):
    """Custom exception raised when no results are found."""
    pass        