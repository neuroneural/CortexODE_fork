#!/bin/bash
. /opt/miniconda3/bin/activate cortexode
cd /cortexode

python eval.py --test_type='init' --data_dir='/speedrun/cortexode-data-rp/test/' --model_dir='/cortexode/ckpts/experiment_4/model/' --init_dir='/cortexode/ckpts/experiment_4/init/test/' --data_name='hcp' --surf_hemi='lh' --tag='exp4' --device='gpu'

