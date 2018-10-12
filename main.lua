--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found here
--  https://github.com/facebook/fb.resnet.torch/blob/master/LICENSE. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  Code modified for Shake-Shake by Xavier Gastaldi
--

require 'torch'
require 'paths'
require 'optim'
require 'nn'
------Shake-Shake------
local json = require 'cjson'
------Shake-Shake------
local DataLoader = require 'dataloader'
local models = require 'models/init'
local Trainer = require 'train'
local opts = require 'opts'
local checkpoints = require 'checkpoints'

-- we don't  change this to the 'correct' type (e.g. HalfTensor), because math
-- isn't supported on that type.  Type conversion later will handle having
-- the correct type.
torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(1)

local opt = opts.parse(arg)
torch.manualSeed(opt.manualSeed)
cutorch.manualSeedAll(opt.manualSeed)

-- Load previous checkpoint, if it exists
local checkpoint, optimState = checkpoints.latest(opt)

-- Create model
local model, criterion = models.setup(opt, checkpoint)

-- Data loading
local trainLoader, valLoader = DataLoader.create(opt)

-- The trainer handles the training loop and evaluation on validation set
local trainer = Trainer(model, criterion, opt, optimState)

if opt.testOnly then
   local top1Err, top5Err = trainer:test(0, valLoader)
   print(string.format(' * Results top1: %6.3f', top1Err))
   return
end

------Shake-Shake------
local filename = tostring(opt.irun) .. "_" .. opt.algorithmType .. "_Wide" .. tostring(opt.widenFactor) .. "_Te" .. tostring(opt.Te) ..
        "_LR" .. tostring(opt.LR) .. "_LRdec" .. opt.LRdec .. "_WD" .. tostring(opt.weightDecay)  .. ".txt"

local snapshots = io.open(paths.concat(opt.snapshots, filename), 'w')
local logfile = io.open(paths.concat(opt.save, filename), 'w')

local function log(t)
   logfile:write('json_stats: '..json.encode(tablex.merge(t,opt,true))..'\n')
   logfile:flush()
end
------Shake-Shake------

local startEpoch = checkpoint and checkpoint.epoch + 1 or opt.epochNumber
local bestTop1 = math.huge
----- Mixup ------
local epoch_seed = 0
----- Mixup ------

for epoch = startEpoch, opt.nEpochs do
   -- Train for a single epoch
   local trainTop1, trainLoss = trainer:train(epoch, trainLoader, epoch_seed)

   -- Run model on validation set
   local testTop1  = trainer:test(epoch, valLoader)

   --local bestModel = false
   --if testTop1 < bestTop1 then
      --bestModel = true
      --bestTop1 = testTop1
      --print(' * Best model ', testTop1)
   --end
   
   ------ Snapshot Ensemble ------
   if (epoch == 120) or (epoch == 360) or (epoch == 840) or (epoch == 1800) then
      print(('Saving snapshot at epoch %d \n'):format(epoch))
      snapshots:write(('==> Snapshot at epoch %d,  top1: %.3f\n'):format(epoch, testTop1))
      snapshots:flush()
      checkpoints.save(epoch, model, trainer.optimState, false, opt)
   end


   --file:write(string.format('%6.3f\t%6.3f\t%6.3f\t%6.3f\n', epoch, trainLoss, trainTop1, testTop1))
   --file:flush()
   ------Shake-Shake------
   log{
      epoch = epoch,
      trainTop1 = trainTop1,
      testTop1  = testTop1,
      trainLoss = trainLoss,
   }

   epoch_seed = epoch_seed + 500000
   ------Shake-Shake------

   --checkpoints.save(epoch, model, trainer.optimState, bestModel, opt)
end
logfile:close()
checkpoints.save(opt.nEpochs, model, trainer.optimState, true, opt)

------Shake-Shake------
-- The error rate for CIFAR-10 should be the error rate obtained at the end of the last epoch, not the best error rate
-- The fb.resnet line below is only valid for Imagenet experiments
print(string.format(' * Finished top1: %6.3f', testTop1))
------Shake-Shake------
