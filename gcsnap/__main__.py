import os

from gcsnap.rich_console import RichConsole 
from gcsnap.configuration import Configuration 
from gcsnap.targets import Target 
from gcsnap.sequence_mapping_online import SequenceMappingOnline
from gcsnap.assemblies import Assemblies


def main():

    starting_directory = os.getcwd()

    console = RichConsole()
    console.print_title()

    # 1. Parse configuration and arguments
    config = Configuration()
    config.parse_arguments()

    # 2. parse targets
    with console.status('Parsing targets'):
        targets = Target(config)
        targets.run()

    # 3. Iterate over each target list
    for out_label in targets.targets_lists:

        # A. Create working directory
        # TODO: Change if out_label_suffix is still used
        working_dir = '{}/{}'.format(starting_directory, out_label)
        if not os.path.isdir(working_dir):
            os.mkdir(working_dir)
        os.chdir(working_dir)

        targets_list = targets.targets_lists[out_label]

        # B. Map sequences to UniProtKB-AC and NCBI EMBL-CDS
        # TODO: Change to whatever input from Joana regardin the mapping.
        # a). Map all targets to UniProtKB-AC
        mappingA = SequenceMappingOnline(config, targets_list, 'UniProtKB-AC')
        mappingA.run()

        # b) Map all to RefSeq
        mappingB = SequenceMappingOnline(config, mappingA.get_codes(), 'RefSeq')
        mappingB.run()
        # merge them to A (only if A is not nan)
        mappingA.merge_mapping_dfs(mappingB.mapping_df)

        # c). Map all targets to NCBI EMBL-CDS
        mappingC = SequenceMappingOnline(config, mappingA.get_codes(), 'EMBL-CDS')
        mappingC.run()
        # merge the two mapping results dataframes
        mappingA.merge_mapping_dfs(mappingC.mapping_df)

        # create targets and ncbi_columns and log not found targets
        mappingA.finalize()
        mapping = mappingA.get_targets_and_ncbi_codes()       
   
        # C. Download and parse assemblies
        #assemblies = Assemblies(config)





    # 4. 


    console.print_final()


    
if __name__ == '__main__':
    main()