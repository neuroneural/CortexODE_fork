#!/bin/bash
. /opt/miniconda3/bin/activate cortexode
cd /cortexode
python eval.py --test_type='pred' --data_dir='/data/users2/washbee/hcp-plis-subj-reviewpapertest/' --model_dir='/data/users2/washbee/speedrun/CortexODE_fork/ckpts/pretrained/adni/' --result_dir='/data/users2/washbee/speedrun/CortexODE_fork/ckpts/reviewpapertest/' --data_name='adni' --surf_hemi='lh' --tag='pretrained' --solver='euler' --step_size=0.1 --device='gpu'
