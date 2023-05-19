#!/bin/bash
#SBATCH -J uni-ti
#SBATCH -p gpu     #V100
#SBATCH -N 1        
#SBATCH -n 20        
#SBATCH --gres=gpu:1    
#SBATCH -o /public/home/jd_cyz/code/unimatch-cyz/more-scenarios/medical/output/output/output_ti-%j.txt      
#SBATCH -e /public/home/jd_cyz/code/unimatch-cyz/more-scenarios/medical/output/output/error_ti-%j.txt   

cd  $SLURM_SUBMIT_DIR
NP=$SLURM_JOB_GPUS

CUDA_VISIBLE_DEVICES=$NP python unimatch.py \
    --use_TI_loss \
    --TI_weight 1e-4 \
    --config /public/home/jd_cyz/code/unimatch-cyz/more-scenarios/medical/configs/acdc.yaml \
    --labeled-id-path /public/home/jd_cyz/code/unimatch-cyz/more-scenarios/medical/splits/acdc/7/labeled.txt \ 
    --unlabeled-id-path /public/home/jd_cyz/code/unimatch-cyz/more-scenarios/medical/splits/acdc/7/unlabeled.txt \
    --save_path /public/home/jd_cyz/code/unimatch-cyz/more-scenarios/medical/output/exp/split-7-0518-2
