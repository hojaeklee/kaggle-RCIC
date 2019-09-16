#!/bin/bash
# The interpreter used to execute the script

#“#SBATCH” directives that convey submission options:

#SBATCH --job-name=4arc
#SBATCH --mail-type=BEGIN,END
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem-per-cpu=4000m 
#SBATCH --time=96:00:00
#SBATCH --account=girasole
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
# The application(s) to execute along with its input arguments and options:

module load python
module load cudnn
module load cuda/10.1.105 
module load cupti/10.1.105 
python train.py -r saved/models/plates_arc_ResNet152/0912_163041/model_best.pth -fp
