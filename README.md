# pytorch-sync-batchnorm-example

The default behavior of Batchnorm, in Pytorch and most other frameworks, is to compute batch statistics separately for each device. Meaning that, if we use a model with batchnorm layers and train on multiple GPUs, batch statistics will not reflect the whole *batch*; instead, statistics will reflect slices of data passed to each GPU. The intuition is that this may harm model convergence and impact performance. In fact, this performance drop is known to happen for object detection models and GANs.

In order to compute batchnorm statistics across all GPUs, we need to use the synchronized batchnorm module that was recently released by Pytorch. To do so, we need to make some changes to our code. We cannot use `SyncBatchnorm` when using `nn.DataParallel(...)`. `SyncBatchnorm` requires that we use a very specific setting: we need to use `torch.parallel.DistributedDataParallel(...)` with Multi-process single GPU configuration. In other words, we need to launch a separate process for each GPU. Below we show step-by-step how to use `SynchBatchnorm` on a single machine with multiple GPUs.

## Basic Idea

We'll launch one process for each GPU. Our training script will be provided a `rank` argument, which is simply an integer that tells us which process is being launched. `rank 0` is our master. This way we can control what each process do. For example, we may want to print losses and stuff to the console only on the master process.

## Step 1: Parsing the local_rank argument

This argument is how we know what process is being lanched. We can have the arguments to our script as usual, we just need to add an extra to parse `--local_rank`.

```python
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', type=int, default=0)
```


## Step 2: Setting up the process and device

Next we need to init the process, we do this by adding the following code to our script:

```python
torch.cuda.set_device(args.local_rank)

world_size = args.ngpu
torch.distributed.init_process_group(
    'nccl',
    init_method='env://',
    world_size=world_size,
    rank=args.local_rank,
)
```



## Step 3: Converting your model to use torch.nn.SyncBatchNorm

We don't need to change our model, it just stay as it is. We just need to convert regular batchnorm layers to [torch.nn.SyncBatchNorm](https://pytorch.org/docs/master/nn.html#torch.nn.SyncBatchNorm).

```python
net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
```

It is our job to send each model to its device:

```python
device = torch.device('cuda:{}'.format(args.local_rank))
net = net.to(device)
```

Remember that we need to do the same for the inputs of the model. i.e.:

```python
for it, (input, target) in enumerate(self.data_loader):
    input, target = input.to(device), target.to(device)
```

## Step 4: Wraping your model with DistributedDataParallel

The same way we wrapped our models with `DataParallel`, we need to do same but with [DistributedDataParallel](https://pytorch.org/docs/master/nn.html#distributeddataparallel).

```python
 net = torch.nn.parallel.DistributedDataParallel(
    net,
    device_ids=[args.local_rank],
    output_device=args.local_rank,
)
```

## Step 5: Adapting your DataLoader

Since we are going to launch multiple processes, we need to take care of the portion of the data provided to each process. This is very simple, assuming you have your Dataset already implemented.

```python
sampler = torch.utils.data.distributed.DistributedSampler(
    dataset,
    num_replicas=config.ngpu,
    rank=local_rank,
)
data_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=8,
    pin_memory=True,
    sampler=sampler,
    drop_last=True,
)
```

## Step 6: Launching the processes

After we parse `--local_rank` and take care of what happens with each process, we can launch the processes using [torch.distributed.launch](https://pytorch.org/docs/master/distributed.html#launch-utility) utility. `--nproc_per_node` is the number of GPUs.

```bash
python -m torch.distributed.launch --nproc_per_node=3 distributed_train.py \
--arg1=arg1 --arg2=arg2 --arg3=arg3 --arg4=arg4 --argn=argn
```

`--arg1=arg1 --arg2=arg2 --arg3=arg3 --arg4=arg4 --argn=argn` are just the regular arguments we pass to our training scripts.
