from subprocess import Popen
import os
import sys
import numpy as np
import itertools
import settings
import argparse
import subprocess


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=".")
    parser.add_argument("--dir", type=int, default=0)
    args = parser.parse_args()
    print('*****************************************************************************')
    print(f'run_sbatch_train_model args = {args}')
    module_path = os.path.dirname(os.path.dirname(os.path.abspath(os.path.join('..'))))
    sys.path.append(module_path)
    settings.init()
    arr_n_latent_attribute_categorical = 2 ** np.arange(4, 8)
    arr_reconstruction_penalty = [1e-1, 1e1, 1e2, 1e3]
    arr_unknown_attribute_penalty = [1e-1, 1e1, 1e2, 1e3]
    arr_unknown_attribute_noise_param = [1e-1, 1e1, 1e2, 1e3]

    parms_combos = itertools.product(arr_n_latent_attribute_categorical,
                                     arr_reconstruction_penalty,
                                     arr_unknown_attribute_penalty,
                                     arr_unknown_attribute_noise_param)

    # create log folder for curren run
    if args.dir == 0:  # if this flag is 0 then we wish to create new repeat - so create now folders
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
    else:
        new_dir_name = args.dir

    job_id = None
    for i, (n_latent_attribute_categorical, reconstruction_penalty, unknown_attribute_penalty,
            unknown_attribute_noise_param) in enumerate(parms_combos):
        # check if the log for this attempt exist then don't run it again for this repeat
        index = str(i + 1)
        log_path = f'../logs/{new_dir_name}/train_model-{index}.log'
        if os.path.exists(log_path):
            print(f"log file {log_path} exist so skipping it")
            continue

        # trying to recreate this command: srun --gres=gpu:1,vmem:10g --mem=100g -c2 --time=20:00:00 --pty $SHELL
        cmdline0 = ['sbatch', '--gres=gpu:a5000:1', '--mem=100gb', '-c1', '--time=3:00:00', '--killable',
                    f'--priority={index}',
                    '--parsable',
                    f"--output='../logs/{new_dir_name}/train_model-{index}.log'",
                    f'--job-name=train-{new_dir_name}-{index}'
                ]
        if job_id is not None:
            cmdline0.append(f'--dependency=afterok:<{str(job_id)}>')

        cmdline1 = [
                    'run_sbatch_train.sh',
                    str(n_latent_attribute_categorical),
                    str(reconstruction_penalty),
                    str(unknown_attribute_penalty),
                    str(unknown_attribute_noise_param),
                    str(i + 1),
                    str(new_dir_name)
                    ]
        cmdline = np.concatenate((cmdline0, cmdline1))
        print(' '.join(cmdline))
        process = Popen(cmdline, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        stdout_list = stdout.decode('utf-8').strip().split()
        print(f"stdout_list = {stdout_list}")
        job_id = stdout_list[-1]
        print(f'jobid = {job_id}')
