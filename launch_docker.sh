docker run --gpus '"device=2"'  --name=em2 \
				 -u $(id -u):$(id -g) \
				 --shm-size 64G \
				 --log-driver=none \
				 --rm -v /scratch/eduard.martynov/datasets/:/workspace/datasets \
				 -v /scratch/eduard.martynov/bclf_4/:/workspace/bclf \
				 -it bclf
