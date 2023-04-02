from subprocess import Popen
import os
import sys
import numpy as np
import itertools
import settings

module_path = os.path.dirname(os.path.dirname(os.path.abspath(os.path.join('..'))))
sys.path.append(module_path)

settings.init()

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

# create log folder for curren run
# curren_dirs_names = [name for name in os.listdir(settings.SAVE_DIR) if os.path.isdir(name) and str(name).isnumeric()]
curren_dirs_names = np.array([name for name in os.listdir(settings.SAVE_DIR) if str(name).isnumeric()], dtype=int)
if len(curren_dirs_names) > 0:
    print(f"curren_dirs_names = {curren_dirs_names}")
    max_folder_name = curren_dirs_names.max()
    new_dir_name = str(max_folder_name + 1)
else:
    new_dir_name = "1"

for dir_path in [settings.SAVE_DIR, settings.FIG_DIR, settings.LOGS_DIR]:
    if not os.path.exists(dir_path + new_dir_name):
        os.makedirs(dir_path + new_dir_name)

settings.init_adata(new_dir_name)

for i, (n_latent_attribute_categorical, reconstruction_penalty, unknown_attribute_penalty, unknown_attribute_noise_param) in enumerate(parms_combos):
    # trying to recreate this command: srun --gres=gpu:1,vmem:10g --mem=100g -c2 --time=20:00:00 --pty $SHELL
    cmdline0 = ['sbatch', '--gres=gpu:a5000:1', '--mem=100gb', '-c1', '--time=10:00:00',
                f'--output=../logs/{new_dir_name}/train_model-{i}.log',
                f'--job-name=train-{i}',
                'run_sbatch_train.sh',
                str(n_latent_attribute_categorical),
                str(reconstruction_penalty),
                str(unknown_attribute_penalty),
                str(unknown_attribute_noise_param),
                str(i+1),
                str(new_dir_name)
                ]
    print(' '.join(cmdline0))
    Popen(cmdline0)
