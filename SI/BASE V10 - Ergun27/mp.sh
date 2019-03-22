#!/bin/bash
#SBATCH --account=def-amrowe
#SBATCH --time=00:31:00
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --mem=100mb
#SBATCH --output=./%x-%j.out

sleep 100
sbatch --array=0-399 mp_R1.sh

sleep 100
sbatch --array=0-399 mp_R2.sh

sleep 100
sbatch --array=0-399 mp_R3.sh

sleep 100
sbatch --array=0-399 mp_R4.sh

sleep 100
sbatch --array=0-399 mp_R5.sh

sleep 100
sbatch --array=0-399 mp_R6.sh
