import ray

ray.init()
EPOCH = 20
import time

@ray.remote
def do_epoch_dummy(partition):
    time.sleep(5)
    for epoch in partition:
        print(epoch)

@ray.remote(num_gpus=1)
def use_gpu():
    print("ray.get_gpu_ids(): {}".format(ray.get_gpu_ids()))
    print("CUDA_VISIBLE_DEVICES: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))

epoch = 0

import math
iterator = range(EPOCH)
NUM_GPU = 8
work_per_gpu = math.ceil(len(iterator) / NUM_GPU)
i = 0
partitions = []

while i * work_per_gpu < len(iterator):
    partitions.append(iterator[i*work_per_gpu: (i+1)*work_per_gpu])
    i += 1

futures = [do_epoch_dummy.remote(p) for p in partitions]
gutures = [use_gpu.remote() for p in partitions]

ray.get(futures)
ray.get(gutures)
