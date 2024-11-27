import os
import shutil

from gcsnap.rich_console import RichConsole 
from gcsnap.configuration import Configuration 
from gcsnap.timing import Timing
from gcsnap.targets import Target 
from gcsnap.mapping import SequenceMapping
from gcsnap.assemblies import Assemblies
from gcsnap.genomic_context import GenomicContext
from gcsnap.sequences import Sequences
from gcsnap.families import Families
from gcsnap.families_functions_structures import FamiliesFunctionsStructures
from gcsnap.operons import Operons
from gcsnap.taxonomy import Taxonomy
from gcsnap.tm_segments import TMsegments
from gcsnap.figures import Figures
from gcsnap.utils import CustomLogger
from gcsnap.parallel_tools import ParallelTools

def main():
    """
    Main function to run the GCsnap pipeline:
    A. Parse configuration file and arguments, initialize the RichConsole and Timing.
    B. Parse targets.
    C. Iterate over each element in target list.
        1. Prework.
        2. Block 'Collect'.
            - a) Map sequences to UniProtKB-AC and NCBI EMBL-CDS.
            - b) Find assembly accession, download and parse assemblies.
            - c) Add sequence information to flanking genes.
        3. Block 'Find families'.
            - a) Find and add protein families.
        4. Block 'Annotate'.
            - a) Add functions and structures to families.
            - b) Find and add operons.
            - c) Get taxonomy information.
            - d) Annotate transmembrane segments.
        5. Produce genomic context figures.
        6. Write output to summary file.
        7. Wrap up.        
    """    

    starting_directory = os.getcwd()
    # Initial logging configuration
    CustomLogger.configure_loggers()

    console = RichConsole('base')
    console.print_title()

    # A. Parse configuration and arguments
    config = Configuration()
    config.parse_arguments()

    # start timing
    timing = Timing()
    t_all = timing.timer('All steps 0-10')

    t_parse = timing.timer('Step 0: Parse Targets')
    # B. parse targets
    targets = Target(config)
    targets.run()
    t_parse.stop()

    # Initialize parallel processes. When Dask is requested, the cluster is set up here.
    parallel = ParallelTools(config)

    # C. Iterate over each element in target list
    # each element is a dictionary with the label as key and the list of targets as value
    for out_label in targets.get_targets_dict():
        # all execution conditions depending on arguments from CLI or config.yaml
        # are handled in the classes themselves

        # 1. Prework 
        working_dir = os.path.join(starting_directory, out_label)
        if not os.path.isdir(working_dir):
            os.mkdir(working_dir)
        os.chdir(working_dir)
        # Configure logger for the current iteration
        CustomLogger.configure_iteration_logger(out_label, starting_directory)
        targets_list = targets.get_targets_dict().get(out_label)
        # set outlabel in console for printing
        RichConsole.set_out_label(out_label)
        if len(targets_list) < 2:
            console.print_warning('GCsnap was asked to analyze only one target')
            console.print_skipped_step('Skipping target {}'.format(out_label))
            continue
        else:
            console.print_working_on('Task {} with {} targets'.format(
                out_label,len(targets_list)))
        # write configuration to log file
        config.write_configuration_yaml_log('input_arguments.log')
        # Datastructure to store all information
        gc = GenomicContext(config, out_label)
        # add targets to genomic context
        gc.curr_targets = targets_list        


        #  2. Block 'Collect'
        t_collect = timing.timer('Step 1: Collecting the genomic contexts')
        # a) Map sequences to UniProtKB-AC and NCBI EMBL-CDS
        t_mapping = timing.timer('Step 1a: Mapping')
        # Map all targets to UniProtKB-AC
        mapping = SequenceMapping(config, targets_list)
        mapping.run()
        targets_and_ncbi_codes = mapping.get_targets_and_ncbi_codes()  
        t_mapping.stop()

        # b). Find assembly accession, download and parse assemblies
        t_assemblies = timing.timer('Step 1b: Assemblies')
        assemblies = Assemblies(config, targets_and_ncbi_codes)
        assemblies.run()
        gc.update_syntenies(assemblies.get_flanking_genes())         
        t_assemblies.stop()

        # c). Add sequence information to flanking genes
        t_sequences = timing.timer('Step 1c: Sequences')        
        sequences = Sequences(config, gc)
        sequences.run()
        gc.update_syntenies(sequences.get_sequences())
        gc.write_syntenies_to_json('genomic_context_information.json')        
        t_sequences.stop()

        t_collect.stop()

        if not config.arguments['collect_only']['value']:

            # 3. Block 'Find families'
            t_family = timing.timer('Step 2: Finding protein families')
            families = Families(config, gc, out_label)
            families.run()
            gc.update_syntenies(families.get_families())
            gc.create_and_write_families_summary()
            t_family.stop() 

            # 4. Block 'Annotate'
            # a) Add functions and structures to families
            t_annotate_families = timing.timer('Step 3: Annotating functions and structures')
            # execution conditions handeled in the class
            ffs = FamiliesFunctionsStructures(config, gc)
            ffs.run()
            gc.update_families(ffs.get_annotations_and_structures())
            gc.write_families_to_json('protein_families_summary.json')
            t_annotate_families.stop()

            # b) Find and add operons        
            t_operons = timing.timer('Step 4-5: Finding operon/genomic_context')
            operons = Operons(config, gc, out_label)
            operons.run()
            gc.update_syntenies(operons.get_operons())
            gc.create_and_write_operon_types_summary()
            gc.find_most_populated_operon_types()   
            t_operons.stop()

            # c) Get taxonomy information
            t_taxonomy = timing.timer('Step 6: Mapping taxonomy')
            taxonomy = Taxonomy(config, gc)
            taxonomy.run()
            gc.update_taxonomy(taxonomy.get_taxonomy())
            gc.write_taxonomy_to_json('taxonomy.json')     
            t_taxonomy.stop()                  

            # d) Annotate TM   
            t_tm = timing.timer('Step 7: Finding ALL proteins with transmembrane segments')
            tm = TMsegments(config, gc, out_label)
            tm.run()
            gc.update_syntenies(tm.get_annotations())
            t_tm.stop()     

            # 5. Produce genomic context figures
            t_figures = timing.timer('Step 8-9: Producing figures')
            figures = Figures(config, gc, out_label, starting_directory)      
            figures.run()

            t_figures.stop()             

            # 6. Write output to summary file
            t_output = timing.timer('Step 10: Write output')
            gc.write_summary_table('{}_summary_table.tab'.format(out_label))

            gc.write_families_to_json('protein_families_summary.json')   

        else:
            console.print_skipped_step('GCsnap was asked to collect genomic context only. Will not proceed further.')
            t_output = timing.timer('Step 10: Write output')

        # 7. Wrap up
        gc.write_syntenies_to_json('all_syntenies.json')
        # log to both loggers (one part of console, one via method)
        CustomLogger.log_to_iteration('Successfully finished task {} with {} targets.'.format(
            out_label,len(targets_list))) 
        console.print_done('Task {} with {} targets'.format(out_label,len(targets_list)))
        t_output.stop()

        # # copy log file to working direcotry
        # shutil.copy(os.path.join(starting_directory,'gcsnap.log'), os.getcwd())

    t_all.stop()
    timing.to_csv('timing.csv')
    console.print_final()
    
if __name__ == '__main__':
    main()