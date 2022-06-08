# birdclef-2022-3rd-place-solution

## Hardware
Deep Learning Server

CPU : Intel(R) Core(TM) i9-10940X CPU @ 3.30GHz 

GPU : 1x3090

RAM : 64Gb

## Software
Python 3.8.10

CUDA: 11.6

cudnn version: 8.0.5.39

NVIDIA Driver Version: 510.73.05

## Setup env
```
cd docker
docker build . -t bclf
cd ..
```

## Launch env
Inside "launch_docker.sh"
```
docker run --gpus '"device=2"'  --name=em2 \
				 -u $(id -u):$(id -g) \
				 --shm-size 64G \
				 --log-driver=none \
				 --rm -v /scratch/eduard.martynov/datasets/:/workspace/datasets \
				 -v /scratch/eduard.martynov/bclf_4/:/workspace/bclf \
				 -it bclf
```
you should change

* --gpus: device id you want to use, for example 0

 change the highlighted part to where competition dataset is stored
* -v **/scratch/eduard.martynov/datasets/**:/workspace/datasets

change the highlighted part to this directory
* -v **/scratch/eduard.martynov/bclf_4/**:/workspace/bclf 

```
bash launch_docker.sh
```
## How to train models
This scripts goes into saved sources, trains the models and copies them into 'save_dir' provided inside 'walk_and_train.sh' script
```
bash launch_docker.sh
cd ./bclf/
python3 walk_and_train.py --save-dir /workspace/bclf/reproduced_experiments \
                          --exp-dir ./experiments
```

## How to do inference
This script does the inference on directory **'inference-dir'**, using saved mopdels from 'experiment-dir', 'exp' and fold 'fold', 
also you can specify per-bird threshold by 'thresholds' value, 

For all experiments you should provide 'cfg' file:

For each one it's "base_config.yaml" except "new-metrics-exp-23", for this experiment you have to provide "tf_effnet_v2_s_in21k.yaml"

'num-workers', 'batch-size' variables are used to increase inference speed

```
bash launch_docker.sh
cd ./bclf/
python3 inference_on_dir.py --experiment-dir ./reproduced_experiments \
                            --exp start_fold4_exp17_full \
                            --inference-dir /workspace/datasets/bird-clef/test_soundscapes/ \
                            --thresholds 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 \
                            --num-workers 16 \
                            --batch-size 32 \
                            --fold 4 \
                            --cfg base_config.yaml                            
```
