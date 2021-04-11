#!/bin/bash
# Created by the University of Melbourne job script generator for SLURM
# Wed Mar 31 2021 13:15:04 GMT+1100 (Australian Eastern Daylight Time)

# Partition for the job:
#SBATCH --partition=physical

# The name of the job:
#SBATCH --job-name="MelbTwitter_1_8"

# The project ID which this job should run under:
#SBATCH --account="COMP90024"

#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --time=0-0:10:00

# check that the script is launched with sbatch
if [ "x$SLURM_JOB_ID" == "x" ]; then
   echo "You need to submit your job to the queuing system with sbatch"
   exit 1
fi

# Run the job from the directory where it was launched (default)

# The modules to load:
module load gcc/8.3.0 openmpi/3.1.4 python/3.7.4

echo "nodes=1 ntasks=8"
# The job command(s):
echo "tinyTwitter output:"
time mpirun python3 test.py melbGrid.json AFINN.txt tinyTwitter.json

echo "smallTwitter output:"
time mpirun python3 test.py melbGrid.json AFINN.txt smallTwitter.json

echo "bigTwitter output:"
time mpirun python3 test.py melbGrid.json AFINN.txt bigTwitter.json
