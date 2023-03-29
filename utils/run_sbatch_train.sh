#!/bin/bash

PROJECT_DIR='/cs/usr/bar246802/bar246802/SandBox2023/biolord_immune_bcells/utils'

n_latent_attribute_categorical=$1
reconstruction_penalty=$2
unknown_attribute_penalty=$3
unknown_attribute_noise_param=$4
id_=$5

module load cuda/11.3
source /cs/usr/bar246802/bar246802/SandBox2023/bioLordVenv/bin/activate

python3 ${PROJECT_DIR}/train_model.py --n_latent_attribute_categorical ${n_latent_attribute_categorical} --reconstruction_penalty ${reconstruction_penalty} --unknown_attribute_penalty ${unknown_attribute_penalty} --unknown_attribute_noise_param ${unknown_attribute_noise_param} --id_ ${id_}