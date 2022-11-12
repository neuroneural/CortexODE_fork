#!/bin/bash
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=30g
#SBATCH -p qTRDGPUH
#SBATCH --gres=gpu:V100:1
#SBATCH -t 1-00:00
#SBATCH -J ctodeval
#SBATCH -e /data/users2/washbee/cortexode/jobs/error%A.err
#SBATCH -o /data/users2/washbee/cortexode/jobs/out%A.out
#SBATCH -A psy53c17
#SBATCH --mail-type=ALL
#SBATCH --mail-user=washbee1@student.gsu.edu
#SBATCH --oversubscribe
#SBATCH --exclude=arctrdgn002


sleep 5s

module load singularity/3.10.2
singularity exec --nv --bind /data,/data/users2/washbee/hcp-plis-subj-pialnn/:/subj,/data/users2/washbee/speedrun/CortexODE_fork:/cortexode /data/users2/washbee/containers/speedrun/cortexODE_sr.sif /cortexode/singularity/eval.sh &

wait

sleep 10s

