import os

from gcsnap.configuration import Configuration
from gcsnap.rich_console import RichConsole
from gcsnap.genomic_context import GenomicContext
from gcsnap.mmseqs_cluster import MMseqsCluster

from gcsnap.utils import processpool_wrapper
from gcsnap.utils import split_dict_chunks

class Families:
    def __init__(self, config: Configuration, gc: GenomicContext, out_label: str):
        self.config = config
        self.cores = config.arguments['n_cpu']['value']

        # set arguments
        self.out_label = out_label
        self.out_dir = os.path.join(os.getcwd(), f'{out_label}_all_against_all_searches')
        self.make_dir(self.out_dir)        
        self.gc = gc
        self.syntenies = gc.get_syntenies()

        self.console = RichConsole()

    def get_families(self) -> dict:
        return self.families_adapted

    def run(self) -> None:
        self.find_cluster()

        with self.console.status('Assigning families to flanking genes'):

            # 1. Add the family to the flanking genes
            # do in parallel, however, each needs the self.cluster_list and self.cluster_order
            # is this might be large, we use as many batches as there are cores
            parallel_args = split_dict_chunks(self.syntenies, self.cores)  
            # a list of tuple[dict, list] is returned
            result_list = processpool_wrapper(self.cores, parallel_args, self.assign_families)
            # combine results
            self.families = {k: v for tup in result_list for k, v in tup[0].items()}
            curr_numbers = [num for tup in result_list for num in tup[1]]
            # sort the curr_numbers
            curr_numbers = sorted(list(set(curr_numbers)))

            # 2. adapt the families where its outside possible ranges
            parallel_args = [(sub_dict, curr_numbers) 
                             for sub_dict in split_dict_chunks(self.families, self.cores)  ] 
            dict_list = processpool_wrapper(self.cores, parallel_args, self.adapt_families)
            # combine results
            self.families_adapted = {k: v for sub_dict in dict_list for k, v in sub_dict.items()}  

    def find_cluster(self) -> list:
        # call MMseqsCluster
        cluster = MMseqsCluster(self.config, self.gc, self.out_dir)
        cluster.run()
        self.cluster_list = cluster.get_clusters_list() 
        self.cluster_order = cluster.get_cluster_order()           
        
    def assign_families(self, args: tuple) -> tuple[dict,list]:
        syntenies = args
        # loop over all targets in the chunk
        curr_numbers = []
        for k in syntenies.keys():
            syntenies[k]['flanking_genes']['family'] = []
            for i, ncbi_code in enumerate(syntenies[k]['flanking_genes']['ncbi_codes']):
                protein_name = syntenies[k]['flanking_genes']['names'][i]
                try:
                    protein_family = self.cluster_list[self.cluster_order.index(ncbi_code)]
                except:
                    protein_family = 10000

                if protein_name == 'pseudogene':
                    protein_family = 10000
                if ncbi_code == k:
                    protein_family = max(self.cluster_list)+1

                syntenies[k]['flanking_genes']['families'].append(protein_family)
                curr_numbers.append(protein_family)

                if ncbi_code == syntenies[k]['assembly_id'][0]:
                    syntenies[k]['target_family'] = protein_family 

        return (syntenies , curr_numbers) 
    
    def adapt_families(self, args: tuple) -> dict:
        # TODO: Check what this curr_numbers is
        syntenies, curr_numbers = args
        for k in syntenies.keys():
            for i, _ in enumerate(syntenies[k]['flanking_genes']['ncbi_codes']):
                protein_family = syntenies[k]['flanking_genes']['families'][i]

                if protein_family <= max(self.cluster_list):
                    cluster_range = range(min(self.cluster_list), max(self.cluster_list)+1)
                    if protein_family != cluster_range[curr_numbers.index(protein_family)]:
                        protein_family = cluster_range[curr_numbers.index(protein_family)]
                        syntenies[k]['flanking_genes']['families'][i] = protein_family

                syntenies[k]['target_family'] = syntenies[k]['flanking_genes']['families'][
                    syntenies[k]['flanking_genes']['ncbi_codes'].index(syntenies[k]['assembly_id'][0])]

        return syntenies
    
    def make_dir(self, path: str) -> None:
        if not os.path.isdir(path):
            os.mkdir(path)