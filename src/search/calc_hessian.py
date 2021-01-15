import os
import sys
import glob
import math
import numpy as np
import torch
import json
import codecs
import pickle
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from copy import deepcopy
from numpy import linalg as LA
from torch.autograd import Variable
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, '../darts-minus')

from src import utils
from src.spaces import spaces_dict
from src.search.model_search import Network
from src.search.architect import Architect
from src.search.analyze import Analyzer
from src.search.args import args, helper, beta_decay_scheduler

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log_{}.txt'.format(args.task_id)))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

schedule_of_params = []


def main(primitives):
  if not torch.cuda.is_available() or args.disable_cuda:
    logging.info('no gpu device available or disabling cuda')
    
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  if not args.disable_cuda:
    torch.cuda.set_device(args.gpu)
    logging.info('gpu device = %d' % args.gpu)
    cudnn.benchmark = True
    cudnn.enabled=True
    torch.cuda.manual_seed(args.seed)

  criterion = nn.CrossEntropyLoss()
  if not args.disable_cuda:
    criterion = criterion.cuda()

  model_init = Network(args.init_channels, args.n_classes, args.layers, criterion,
                       primitives, steps=args.nodes, args=args)
  if not args.disable_cuda:
    model_init = model_init.cuda()
  logging.info("param size = %fMB", utils.count_parameters_in_MB(model_init))

  optimizer_init = torch.optim.SGD(
      model_init.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay)

  architect_init = Architect(model_init, args)

  scheduler_init = CosineAnnealingLR(
        optimizer_init, float(args.epochs), eta_min=args.learning_rate_min)

  analyser = Analyzer(args, model_init)
  la_tracker = utils.EVLocalAvg(args.window, args.report_freq_hessian,
                                args.epochs)

  train_queue, valid_queue, train_transform, valid_transform = helper.get_train_val_loaders()

  def valid_generator():
    while True:
      for x, t in valid_queue:
        yield x, t

  valid_gen = valid_generator()
  
  for epoch in range(args.ev_start_epoch-1, args.epochs):
    beta_decay_scheduler.step(epoch)
    logging.info("EPOCH %d SKIP BETA DECAY RATE: %e", epoch, beta_decay_scheduler.decay_rate)
    if (epoch % args.report_freq_hessian == 0) or (epoch == (args.epochs - 1)):
        lr = utils.load_checkpoint(model_init, optimizer_init, None,
                    architect_init, args.save, la_tracker,
                    epoch, args.task_id)
        logging.info("Loaded %d-th checkpoint."%epoch)

        if args.test_infer:
            valid_acc, valid_obj = infer(valid_queue, model_init, criterion)
            logging.info('valid_acc %f', valid_acc)

        if args.compute_hessian:
            input, target = next(iter(train_queue))
            input = Variable(input, requires_grad=False)
            target = Variable(target, requires_grad=False)
            input_search, target_search = next(valid_gen) #next(iter(valid_queue))
            input_search = Variable(input_search, requires_grad=False)
            target_search = Variable(target_search, requires_grad=False)

            if not args.disable_cuda:
                input=input.cuda()
                target=target.cuda(async=True)
                input_search=input_search.cuda()
                target_search=target_search.cuda(async=True)

            if not args.debug:
                H = analyser.compute_Hw(input, target, input_search, target_search,
                                        lr, optimizer_init, False)
                g = analyser.compute_dw(input, target, input_search, target_search,
                                        lr, optimizer_init, False)
                g = torch.cat([x.view(-1) for x in g])

                state = {'epoch': epoch,
                        'H': H.cpu().data.numpy().tolist(),
                        'g': g.cpu().data.numpy().tolist(),
                        #'g_train': float(grad_norm),
                        #'eig_train': eigenvalue,
                        }

                with codecs.open(os.path.join(args.save,
                                            'derivatives_{}.json'.format(args.task_id)),
                                            'a', encoding='utf-8') as file:
                    json.dump(state, file, separators=(',', ':'))
                    file.write('\n')

                # early stopping
                ev = max(LA.eigvals(H.cpu().data.numpy()))
                logging.info('CURRENT EV: %f', ev)

def infer(valid_queue, model, criterion):
    objs = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    model.eval()

    def valid_generator():
        for x, t in valid_queue:
            yield x, t

    valid_gen = valid_generator()

    # for step, (input, target) in enumerate(valid_queue):
    step=0
    for input, target in valid_gen:
        step+=1
        input = Variable(input)
        target = Variable(target)

        if not args.disable_cuda:
            input=input.cuda()
            target=target.cuda(async=True) 

        logits = model(input,0)
        loss = criterion(logits, target)

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        if step % args.report_freq == 0:
            logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
        if args.debug:
            break

    return top1.avg, objs.avg

if __name__ == '__main__':
  space = spaces_dict[args.space]
  main(space)
