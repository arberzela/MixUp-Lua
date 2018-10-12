Training recipes
----------------

### CIFAR-10

To train 26 2x32d Shake-Shake + CutOut + MixUp on CIFAR-10 on 1 GPUs with SGDR:

```bash
th main.lua -dataset cifar10 -nGPU 1 -batchSize 64 -depth 26 -shareGradInput false -optnet true -nEpochs 1800 -netType shakeshake -lrShape cosine -baseWidth 32 -LR 0.1 -forwardShake true -backwardShake true -shakeImage true -Te 120 -Tmult 2 -widenFactor 1 -irun 1 -cutout_half_size 8 -alpha 0.2
```

For models with different number of initial filters change the `-baseWidth` flag:

To run the model on two (or more) GPUs, you will need to use the [`-shareGradInput`](#sharegradinput) flag and scale the batch size and learning rate accordingly:

```bash
th main.lua -dataset cifar10 -nGPU 2 -batchSize 128 -depth 26 -shareGradInput true -optnet false -nEpochs 1800 -netType shakeshake -lrShape cosine -baseWidth 32 -LR 0.2 -forwardShake true -backwardShake true -shakeImage true -Te 120 -Tmult 2 -widenFactor 1 -irun 1 -cutout_half_size 8 -alpha 0.2

```

## Useful flags

For a complete list of flags, run `th main.lua --help`.

### shareGradInput

The `-shareGradInput` flag enables sharing of `gradInput` tensors between modules of the same type. This reduces
memory usage. It works correctly with the included ResNet models, but may not work for other network architectures. See 
[models/init.lua](models/init.lua#L42-L60) for the implementation.

The `shareGradInput` implementation may not work with older versions of the `nn` package. Update your `nn` package by running `luarocks install nn`.

### shortcutType

The `-shortcutType` flag selects the type of shortcut connection. The [ResNet paper](http://arxiv.org/abs/1512.03385) describes three different shortcut types:
- `A`: identity shortcut with zero-padding for increasing dimensions. This is used for all CIFAR-10 experiments.
- `B`: identity shortcut with 1x1 convolutions for increasing dimesions. This is used for most ImageNet experiments.
- `C`: 1x1 convolutions for all shortcut connections.
