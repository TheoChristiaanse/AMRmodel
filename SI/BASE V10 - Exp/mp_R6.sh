#!/bin/bash
#SBATCH --account=def-amrowe
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --mem=1000mb
#SBATCH --job-name=V10_R6
#SBATCH --output=./output/output-%x-%a.out
#SBATCH --open-mode=append

if test -e ./pickleddata/$SLURM_JOB_NAME-$SLURM_ARRAY_TASK_ID; then
  # Rerun the simulation
  module load python/3.7
  stat `which python3.7`
  source ../ENV2/bin/activate
  echo "prog restarted at: `date`"
  mpiexec python ./$SLURM_JOB_NAME.py $SLURM_ARRAY_TASK_ID $SLURM_JOB_NAME
else
  # First Run
  echo "Current working directory: `pwd`"
  echo "Starting run at: `date`"
  echo ""
  echo "Job Array ID / Job ID: $SLURM_ARRAY_JOB_ID / $SLURM_JOB_ID"
  echo "This is job $SLURM_ARRAY_TASK_ID out of $SLURM_ARRAY_TASK_COUNT jobs."
  echo ""
  module load python/3.7
  stat `which python3.7`
  source ../ENV2/bin/activate
  echo "prog started at: `date`"
  mpiexec python ./$SLURM_JOB_NAME.py $SLURM_ARRAY_TASK_ID $SLURM_JOB_NAME
fi

deactivate

if test -e ./pickleddata/$SLURM_JOB_NAME-$SLURM_ARRAY_TASK_ID; then
    # If there is pickeled data available resubmit just this single job.
    sbatch --array=$SLURM_ARRAY_TASK_ID ${BASH_SOURCE[0]}
else
    # If we have steady state there will be no pickled data file
    # at the end of the simulation.
    echo "prog ended at: `date`"
fi
