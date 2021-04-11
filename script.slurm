#!/bin/bash
# Created by the University of Melbourne job script generator for SLURM
# Wed Mar 31 2021 13:15:04 GMT+1100 (Australian Eastern Daylight Time)

# Partition for the job:
#SBATCH --partition=physical

# The name of the job:
#SBATCH --job-name="MelbTwitter"

# The project ID which this job should run under:
#SBATCH --account="COMP90024"

#SBATCH --nodes=2
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --time=0-0:04:00

# check that the script is launched with sbatch
if [ "x$SLURM_JOB_ID" == "x" ]; then
   echo "You need to submit your job to the queuing system with sbatch"
   exit 1
fi

# Run the job from the directory where it was launched (default)

# The modules to load:
module load Python/3.4.3-goolf-2015a

# The job command(s):
mpirun python3 test.py melbGrid.json AFINN.txt tinyTwitter.json
mpirun python3 test.py melbGrid.json AFINN.txt smallTwitter.json
mpirun python3 test.py melbGrid.json AFINN.txt bigTwitter.json