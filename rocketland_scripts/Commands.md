# Commands

## Setup Environment

1. `conda activate param_deepreach`

2. `conda activate /home/somilban/.conda/envs/siren`

3. `export PATH="/home/jason/miniconda3/envs/deepreach/bin:/home/jason/miniconda3/condabin:/usr/local/anaconda3/bin:/usr/local/cuda/bin:/usr/local/anaconda3/bin:/usr/local/cuda/bin:/usr/local/cuda/bin:/home/xingpeng/.vscode-server/bin/74f6148eb9ea00507ec113ec51c489d6ffb4b771/bin/remote-cli:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin"`


## Rocket Landing

1. `CUDA_VISIBLE_DEVICES=0 python ./rocketland_scripts/train_hji_parameterconditioned_simplerocketlanding.py --experiment_name rocket2 --pretrain --diffModel --adjust_relative_grads`

`CUDA_VISIBLE_DEVICES=0 python ./rocketland_scripts/validate_hji_parameterconditioned_simplerocketlanding.py --experiment_name rocket1 --pretrain --diffModel --adjust_relative_grads`

2. `CUDA_VISIBLE_DEVICES=1 python ./air3D_scripts/train_hji_air3D_p1D.py --experiment_name air3D1 --pretrain --diffModel --adjust_relative_grads`

3. `CUDA_VISIBLE_DEVICES=0 python ./tripleVehicle_scripts/train_multivehicle_beta.py --experiment_name multivehicle_beta1 --pretrain --diffModel --adjust_relative_grads`

4. `CUDA_VISIBLE_DEVICES=1 python ./tripleVehicle_scripts/train_multivehicle_collision_NE.py --experiment_name multivehicle_collision_NE1 --pretrain --diffModel --adjust_relative_grads`

5. visualize Pfs:

`CUDA_VISIBLE_DEVICES=0 python ./rocketland_scripts/validate_hji_pfs.py --experiment_name rocket1 --pretrain --diffModel --adjust_relative_grads`

## Lunar Landing
1. `CUDA_VISIBLE_DEVICES=0 python ./rocketland_scripts/train_lunar_landing.py --experiment_name LunarLanding --pretrain --diffModel --adjust_relative_grads`

## trans reach training

1. `CUDA_VISIBLE_DEVICES=0 python ./rocketland_scripts/train_transreach_rocket.py --experiment_name rocket_trans`