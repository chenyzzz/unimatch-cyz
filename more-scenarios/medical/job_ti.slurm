#!/bin/bash
#SBATCH -J uni-ti
#SBATCH -p gpu     #V100
#SBATCH -N 1        
#SBATCH -n 20        
#SBATCH --gres=gpu:1    
#SBATCH -o /public/home/jd_cyz/code/unimatch-cyz/more-scenarios/medical/output/output_ti-%j.txt      
#SBATCH -e /public/home/jd_cyz/code/unimatch-cyz/more-scenarios/medical/output/error_ti-%j.txt   

cd  $SLURM_SUBMIT_DIR
NP=$SLURM_JOB_GPUS

CUDA_VISIBLE_DEVICES=$NP python unimatch.py \
    --use_TI_loss \
    --config /public/home/jd_cyz/code/unimatch-cyz/more-scenarios/medical/configs/acdc.yaml \
<<<<<<< HEAD
    --save_path /public/home/jd_cyz/code/unimatch-cyz/more-scenarios/medical/output/exp/0517
=======
    --save-path /public/home/jd_cyz/code/unimatch-cyz/more-scenarios/medical/output/exp
>>>>>>> 594110686621152920ef76a527c16dad7b933e6c
