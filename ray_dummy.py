import torch
import ray
import os

ray.init()
EPOCH = 20
import time

x = 5

def goo():
    print(x)

@ray.remote
def foo(mult):
    global x
    x = 20 * mult
    goo()

@ray.remote
def do_epoch_dummy(partition):
    time.sleep(5)
    for epoch in partition:
        print(epoch)

# @ray.remote(num_gpus=8)
# def use_gpu(gpu_id):
#     print("ID: {}".format(gpu_id))
#     print("ray.get_gpu_ids(): {}".format(ray.get_gpu_ids()))
#     print("CUDA_VISIBLE_DEVICES: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
#     with torch.cuda.device(gpu_id):
#         a = torch.Tensor([gpu_id]).cuda()
#         b = torch.Tensor([3]).cuda()
#         c = a+b
#     print("{} at {}".format(c, c.device))
#     torch.cuda.empty_cache()

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
# gutures = [use_gpu.remote(e) for e,p in enumerate(partitions)]

#ray.get(futures)
#ray.get(gutures)
ray.get([foo.remote(i) for i in range(1,6)])