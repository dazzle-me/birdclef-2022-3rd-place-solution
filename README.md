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


## Data
Aside from competition data, the solution used following external data:
1) freefield1010 dataset
2) BirdVox-DCASE-20k dataset
3) no-call parts of 2021 training data

Fist two datasets can be downloaded from [here](https://dcase.community/challenge2018/task-bird-audio-detection), the last one can be found [here](https://www.kaggle.com/datasets/christofhenkel/birdclef2021-background-noise)

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
This scripts goes into saved sources from 'exp-dir', trains the models and copies them into 'save-dir'
```
bash launch_docker.sh
cd ./bclf/
python3 walk_and_train.py --save-dir /workspace/bclf/reproduced_experiments \
                          --exp-dir ./experiments
```

## How to do inference
arguments:
* inference-dir: test dir
* experiment-dir: dir with all experiments stored
* exp:  experiment name
* fold:  fold used as validation during training
* thresholds:  per-bird threshold values
* num-workers: number of workers used in DataLoader
* batch_size:  batch size to speed-up inference
* cfg:  always 'base_config.yaml' except for 'new-metrics-exp-23', for this experiment pass 'tf_effnet_v2_s_in21k.yaml'
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
