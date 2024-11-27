# GCsnap2.0 Cluster

This is the implementation of GCsnap2.0 designed to run on sciCORE managed with SLURM under Ubuntu.

GCsnap is a flexible Python-based tool that allows for the interactive comparison of the genomic contexts of protein-coding genes from any genome at any taxonomic level, integrating them with functional and structural information for any of the genes shown. 

GCsnap is not limited to a single input format, can preform batch jobs and accepts protein classification maps. 

All information is stored in detailed, human and machine-readable files, and customable publication-ready figures.


Thank you for using and showing interest on GCsnap!

## Date requirements
As designed to run on a cluster, GCsnap requires the needed data to be locally available.  
The structure looks like this, the source are given below:
```
data-path as defined in config.yaml or via CLI  
  ├── genbank  
  |   └── assembly_summary_genbank.txt   
  │   └── data   
  │       └── GCA_000001405.15_genomic.gff.gz  
  │       └── GCA_000001405.15_protein.faa.gz      
  ├── refseq  
  |   └── assembly_summary_refseq.txt     
  │   └── data   
  │       └── GCF_000001405.38_genomic.gff.gz  
  │       └── GCF_000001405.38_protein.faa.gz      
  ├── mappings  
  │   └── idmapping_selected.tab  
  ├── db  
  │   └── assemblies.db  
  │   └── mappings.db  
  │   └── sequences.db  
  │   └── rankedlineage.dmp  
```

All the scripts to syncronize the data are available in [download_scripts](/download_scripts) folder.  
The scripts to create the databases are available in [creade_db_scripts](/create_db_scripts) folder.  

### Data source
- [GenBank summary](https://ftp.ncbi.nlm.nih.gov/genomes/genbank)  
- [RefSeq summary](https://ftp.ncbi.nlm.nih.gov/genomes/refseq)
- [GenBank data](https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA)
- [RefSeq data](https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF)
- [UniProt mappings](https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/idmapping)
- [Taxonomy Taxdump.tar.gz](https://ftp.ncbi.nlm.nih.gov/pub/taxonomy)



## Dependencies

GCsnap2.0 Cluster is written in Python 3.11. It was tested on Python 3.11. It requires mostly core Python modules and some external packages: 
  - Biopython
  - Bokeh
  - Matplotlib
  - Networkx 
  - PaCMAP (requires Numba which requires NumPy)
  - Scikit-learn
  - Pandas
  - Rich
  - Jinja2
  - mpi4py

For detailed requirements including working versions, check ```pyproject.toml```.

Additionally, GCsnap relies on a local installation of MMseqs.

## Installation

### Installing from Source

Clone the repository with git, create a Conda environment and install:

**Linux on the cluster**  
It was tested with Ubuntu. The problem is that Conda environments created under CentOS do not work on Ubuntu and vice versa.
So we created it on the Login node of sciCORE and **not** on the Schwede worker node.
```
# To download
git clone https://github.com/GCsnap/gcsnap2cluster.git
cd gcsnap2cluster

# To install
conda create -n GCsnapC -c conda-forge -c bioconda gcc=14.1 mpich=4.2.2 python=3.11 mmseqs2
conda activate GCsnapC
pip install .
```

## Allowed inputs

GCsnap takes as main input a list of sequence identifiers, which can be in **Entrez, UniprotKB, UniRef, GeneID, and ENSEMBLE ID formats, or a mix**. These identifiers can be given as:
  - a text file, where each is in a different line
  - a fasta file, where the sequence header starts with the sequence identifier
  - a sequences cluster file in CLANS format
  - direct input in the terminal as a space-separated list
  
## Usage

To execute it on a cluster, please refer to the scripts in [SLURM scripts](/slurm_scripts) folder. However, it is also working from terminal in a similar way as GCsnap2.0 Desktop version but with the use of srun:
```
srun --mpi=pmi2 -N 2 --ntasks-per-node=4 python -m mpi4py.futures ./gcsnap/__main__.py --n-nodes 2 --n-cpu-per-node 4 --targets ./examples/target_sequences_10.txt
```
