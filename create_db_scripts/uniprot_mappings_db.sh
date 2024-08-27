#!/bin/bash

## Load modlues
ml Python/3.10.4-GCCcore-11.3.0
export PYTHONPATH=$PYTHONPATH:/scicore/home/schwede/kruret00/MT/GCsnap2

## script to to call run.job which calls db_create_uniprot_mappings.py to create assembly and sequence db for GCsnap2.0
## Author: Reto Krummenacher

# paths
path=/scicore/home/schwede/GROUP/gcsnap_db/
exp_path=${path}create_db_scripts/
gcsnap_path=/scicore/home/schwede/kruret00/MT/GCsnap2/gcsnap/

# arguments
nodes=1

ident=create_db_uniport

sbatch --export=ALL,gcsnap_path=${gcsnap_path} --job-name=${ident} --nodes=${nodes} --ntasks=1 --output=${exp_path}${ident}.out ${exp_path}uniprot_mappings_db.job

