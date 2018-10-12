--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  Multi-threaded data loader
--

local datasets = require 'datasets/init'
local Threads = require 'threads'
Threads.serialization('threads.sharedserialize')

local M = {}
local DataLoader = torch.class('resnet.DataLoader', M)

function DataLoader.create(opt)
   -- The train and val loader
   local loaders = {}

   for i, split in ipairs{'train', 'val'} do
      local dataset = datasets.create(opt, split)
      loaders[i] = M.DataLoader(dataset, opt, split)
   end

   return table.unpack(loaders)
end

function DataLoader:__init(dataset, opt, split)
   local manualSeed = opt.manualSeed
   local function init()
      require('datasets/' .. opt.dataset)
   end
   local function main(idx)
      if manualSeed ~= 0 then
         torch.manualSeed(manualSeed + idx)
      end
      torch.setnumthreads(1)
      _G.dataset = dataset
      _G.preprocess = dataset:preprocess()
      return dataset:size()
   end

   local threads, sizes = Threads(opt.nThreads, init, main)
   self.nCrops = (split == 'val' and opt.tenCrop) and 10 or 1
   self.threads = threads
   self.__size = sizes[1][1]
   self.batchSize = math.floor(opt.batchSize / self.nCrops)
   self.opt = opt
   local function getCPUType(tensorType)
      if tensorType == 'torch.CudaHalfTensor' then
         return 'HalfTensor'
      elseif tensorType == 'torch.CudaDoubleTensor' then
         return 'DoubleTensor'
      else
         return 'FloatTensor'
      end
   end
   self.cpuType = getCPUType(opt.tensorType)
end

function DataLoader:size()
   return math.ceil(self.__size / self.batchSize)
end

function DataLoader:runTrain(epoch_seed)
   local threads = self.threads
   local size, batchSize = self.__size, self.batchSize
   local alpha = self.opt.alpha
   local perm = torch.randperm(size)
   local epoch_seed = epoch_seed

   local idx, sample, sample1, sample2 = 1, nil, nil, nil
   local function enqueue()
      while idx <= size and threads:acceptsjob() do
         local indices = perm:narrow(1, idx, math.min(batchSize, size - idx + 1))
         threads:addjob(
            function(indices, nCrops, cpuType)
               local randomkit = require 'randomkit'
               local sz = indices:size(1)
               local perm1 = torch.randperm(sz) -- shuffle the current batch for mixup
               local batch, imageSize, lam
               local target_a = torch.IntTensor(sz)
               local target_b = torch.IntTensor(sz)
    
               lam = randomkit.beta(torch.Tensor(1), alpha, alpha)[1]

               for i, idx in ipairs(indices:totable()) do
                  local seed1 = epoch_seed + idx
                  local seed2 = epoch_seed + indices[perm1[i]]
                  local sample1 = _G.dataset:get(idx)
                  local input1 = _G.preprocess(sample1.input, seed1)
                  local sample2 = _G.dataset:get(indices[perm1[i]])
                  local input2 = _G.preprocess(sample2.input, seed2)
                  if not batch then
                     imageSize = input1:size():totable()
                     if nCrops > 1 then table.remove(imageSize, 1) end
                     batch = torch[cpuType](sz, nCrops, table.unpack(imageSize))
                  end
                  local input = lam * input1 + (1 - lam) * input2
                  batch[i]:copy(input)
                  target_a[i] = sample1.target
                  target_b[i] = sample2.target
               end
               collectgarbage()
               return {
                  input = batch:view(sz * nCrops, table.unpack(imageSize)),
                  target_a = target_a,
                  target_b = target_b,
                  lam = lam,
               }
            end,
            function(_sample_)
               sample = _sample_
            end,
            indices,
            self.nCrops,
            self.cpuType
         )
         idx = idx + batchSize
      end
   end

   local n = 0
   local function loop()
      enqueue()
      if not threads:hasjob() then
         return nil
      end
      threads:dojob()
      if threads:haserror() then
         threads:synchronize()
      end
      enqueue()
      n = n + 1
      return n, sample
   end

   return loop
end

function DataLoader:runTest()
    local threads = self.threads
    local size, batchSize = self.__size, self.batchSize
    local perm = torch.randperm(size)

    local idx, sample = 1, nil
    local function enqueue()
        while idx <= size and threads:acceptsjob() do
            local indices = perm:narrow(1, idx, math.min(batchSize, size - idx + 1))
            threads:addjob(
                function(indices, nCrops, cpuType)
                    local sz = indices:size(1)
                    local batch, imageSize
                    local target = torch.IntTensor(sz)
                    for i, idx in ipairs(indices:totable()) do
                        local sample = _G.dataset:get(idx)
                        local input = _G.preprocess(sample.input)
                        if not batch then
                            imageSize = input:size():totable()
                            if nCrops > 1 then table.remove(imageSize, 1) end
                            batch = torch[cpuType](sz, nCrops, table.unpack(imageSize))
                        end
                        batch[i]:copy(input)
                        target[i] = sample.target
                    end
                    collectgarbage()
                    return {
                        input = batch:view(sz * nCrops, table.unpack(imageSize)),
                        target = target,
                    }
                end,
                function(__sample__)
                    sample = __sample__
                end,
                indices,
                self.nCrops,
                self.cpuType
            )
            idx = idx + batchSize
        end
    end

    local n = 0
    local function loop()
        enqueue()
        if not threads:hasjob() then
            return nil
        end
        threads:dojob()
        if threads:haserror() then
            threads:synchronize()
        end
        enqueue()
        n = n + 1
        return n, sample
    end

    return loop
end

return M.DataLoader
