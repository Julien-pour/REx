import argparse
import os
import subprocess
# arg train
parser = argparse.ArgumentParser(description="Argument parsing for ARC experiment")
parser.add_argument("--dev", action=argparse.BooleanOptionalAction, help="Development mode")
parser.add_argument("--long", action=argparse.BooleanOptionalAction, help="long mode 100h instead of 20h")
parser.add_argument("--medium", action=argparse.BooleanOptionalAction, help="medium mode 40h instead of 20h")
parser.add_argument("--hour", type=int, default=20, help="Number of hours")
parser.add_argument("--n_gpu", type=int, default=1, help="Number of hours")
parser.add_argument("--model_path", type=str, help="Number of hours")
parser.add_argument("--h100", action=argparse.BooleanOptionalAction, default =True, help="medium mode 40h instead of 20h")
parser.add_argument('--max-steps', type=int, default=300)
args = parser.parse_args()


def generate_slurm_script(args,job_name):

    if args.dev:
        if args.h100:
            dev_script = "#SBATCH --qos=qos_gpu_h100-dev"
        
        else:
            dev_script = "#SBATCH --qos=qos_gpu_a100-dev"
    else:        
        dev_script = ""

    h = '2' if args.dev else str(args.hour)
    
    if args.long:
        h = '99'
        if args.hour !=20:
            h = str(args.hour)
            
        elif args.h100:
            dev_script = "#SBATCH --qos=qos_gpu_h100-t4"

    if args.medium:
        if args.h100:
            dev_script = "#SBATCH --qos=qos_gpu_h100-t4"

    if args.h100:
        account = "imi@h100"
        c="h100"
        n_cpu = min(int(args.n_gpu * 24),96)
        module_load = "module load arch/h100"
    else:
        account = "imi@a100"
        c="a100"
        n_cpu = min(int(args.n_gpu * 8),64)
        module_load = "module load arch/a100"
    script = f"""#!/bin/bash
#SBATCH --account={account}
#SBATCH -C {c}
#SBATCH --job-name={job_name}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:{args.n_gpu}
#SBATCH --cpus-per-task={n_cpu}
{dev_script}
#SBATCH --hint=nomultithread
#SBATCH --time={h}:00:00
#SBATCH --output=./out/{job_name}-%A.out
#SBATCH --error=./out/{job_name}-%A.out
# set -x

export TMPDIR=$JOBSCRATCH
module purge
module load arch/h100
module load python/3.10.4

ulimit -c 0
limit coredumpsize 0
export CORE_PATTERN=/dev/null
conda deactivate
conda activate acr
module load cuda/12.4.1
export TORCH_CUDA_ARCH_LIST="9.0"

"""

    return script



script_inference = """

cd /lustre/fsn1/projects/rech/imi/uqv82bm/REx/
export OPENAI_API_KEY="None"
python -m acr.run --domain apps --apps-difficulty comp --scheduler rex --rex-constant 20 --llm-seed 0 --llm-model {model_path} --n_gpu {n_gpu} --max-steps {max_steps}

"""
script_inference = script_inference.format(model_path=args.model_path, n_gpu=args.n_gpu, max_steps=args.max_steps)
model_name=args.model_path.split("/")[-1]
job_name="REX_"+model_name
full_script = generate_slurm_script(args,job_name=job_name) + script_inference

slurmfile_path = f'run_{job_name}.slurm'

with open(slurmfile_path, 'w') as f:
    f.write(full_script)

subprocess.call(f'sbatch {slurmfile_path}', shell=True)

# del slurmfile_path
os.remove(slurmfile_path)

