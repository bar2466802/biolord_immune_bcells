from subprocess import Popen
import os
import sys
import numpy as np
import itertools


module_path = os.path.dirname(os.path.dirname(os.path.abspath(os.path.join('..'))))
sys.path.append(module_path)

DATA_DIR = "../data/"
SAVE_DIR = "../output/"
FIG_DIR = "../figures/"
LOGS_CSV = SAVE_DIR + "trained_model_scores.csv"

arr_n_latent_attribute_categorical = 2 ** np.arange(4, 8)
arr_reconstruction_penalty = [1e-1, 1e1, 1e2, 1e3]
arr_unknown_attribute_penalty = [1e-1, 1e1, 1e2, 1e3]
arr_unknown_attribute_noise_param = [1e-1, 1e1, 1e2, 1e3]

# arr_n_latent_attribute_categorical = np.concatenate((np.arange(3, 5, 1), np.arange(5, 31, 5)))
# arr_reconstruction_penalty = [1e1, 1e2, 1e3]
# arr_unknown_attribute_penalty = [1e-2, 1e-1, 1e1]
# arr_unknown_attribute_noise_param = [1e-2, 1e-1, 1e1]

parms_combos = itertools.product(arr_n_latent_attribute_categorical,
                                 arr_reconstruction_penalty,
                                 arr_unknown_attribute_penalty,
                                 arr_unknown_attribute_noise_param)

for i, (n_latent_attribute_categorical, reconstruction_penalty, unknown_attribute_penalty, unknown_attribute_noise_param) in enumerate(parms_combos):
    # trying to recreate this command: srun --gres=gpu:1,vmem:10g --mem=100g -c2 --time=20:00:00 --pty $SHELL
    cmdline0 = ['sbatch', '--gres=gpu:a5000:1', '--mem=100gb', '-c1', '--time=10:00:00',
                f'--output=../logs/train_model-{i}.log',
                f'--job-name=train-{i}',
                'run_sbatch_train.sh',
                str(n_latent_attribute_categorical),
                str(reconstruction_penalty),
                str(unknown_attribute_penalty),
                str(unknown_attribute_noise_param),
                str(i+1)
                ]
    print(' '.join(cmdline0))
    Popen(cmdline0)
