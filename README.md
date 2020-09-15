# GCsnap

GCsnap is a flexible Python-based tool that allows for the interactive comparison of the genomic contexts of protein-coding genes from any genome at any taxonomic level, integrating them with functional and structural information for any of the genes shown. 

![figure1](https://github.com/JoanaMPereira/GCsnap/blob/master/examples/Fig1.png)

By connecting the output to different protein databases, the user can navigate through the different genomic contexts from a simple interactive platform, facilitating the further analysis of the contexts found. GCsnap is not limited to a single input format, can preform batch jobs and accepts protein classification maps. 

All information is stored in detailed, human and machine-readable files, and customable publication-ready figures.

## Dependencies

GCsnap is written in Python 3.7.4 and requires mostly core Python modules. Only three external packages are required: 
  - Biopython
  - Bokeh
  - Networkx 

Additionally, GCsnap relies on a local installation of BLASTp and PsiBlast (versions 2.4.0+ and above). 

## Allowed inputs

GCsnap takes as main input a list of sequence identifiers, which can be in **Entrez, UniprotKB, and UniRef formats, or a mix**. These identifiers can be given as:
  - a text file, where each is in a different line
  - a fasta file, where the sequence header starts with the sequence identifier
  - a sequences cluster file in CLANS format
  - direct input in the terminal as a space-separated list
  
## Usage
