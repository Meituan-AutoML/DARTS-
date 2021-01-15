import sys
import os
sys.path.append('./')

import csv
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import cm
import seaborn as sns
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dset
import json

from src import utils
from src.spaces import spaces_dict
from src.search.model_search import Network
from src.landscape.plot_2D import plot_2D

from src.search.args import args, beta_decay_scheduler
if not args.show:
  matplotlib.use('Agg')

args.xmin, args.xmax, args.xnum = [float(a) for a in args.x.split(':')]
args.ymin, args.ymax, args.ynum = [float(a) for a in args.y.split(':')]

# args.n_classes = 10
torch.backends.cudnn.deterministic = True

def load_alpah(alpha_file):
    print("Loading alpha from %s"%alpha_file)
    alphas_value = json.load(open(alpha_file,'r'))
    return [torch.from_numpy(np.array(alphas_value['alphas_normal'])), torch.from_numpy(np.array(alphas_value['alphas_reduce']))]

def create_random_direction(alphas):
    """
        Setup a random (normalized) direction with the same dimension as
        the weights or states.

        Args:
          net: the given trained model
          dir_type: 'weights' or 'states', type of directions.
          ignore: 'biasbn', ignore biases and BN parameters.
          norm: direction normalization method, including
                'filter" | 'layer' | 'weight' | 'dlayer' | 'dfilter'

        Returns:
          direction: a random direction with the same dimension as weights or states.
    """
    # random direction
    direction =  [torch.randn(a.size()) for a in alphas]
    return direction

def obtain_grad_direction(model, valid_queue):
  direction = [0. for a in model.arch_parameters()]
  alphas = model.arch_parameters()
  for step, (input, target) in enumerate(valid_queue):
    input = Variable(input, volatile=True)
    target = Variable(target, volatile=True)
    if not args.disable_cuda:
      input = input.cuda()
      target = target.cuda(async=True)

    logits = model(input, 0)
    loss = criterion(logits, target)
    grads_alphas = torch.autograd.grad(loss, alphas, grad_outputs=None,
                       allow_unused=True,
                       retain_graph=None,
                       create_graph=False)
    direction = [d+grad for d, grad in zip(direction, grads_alpha)]

  direction = [d/len(valid_queue) for d in direction]
  return direction

def norm_direction(direction, norm, alphas):
  if norm == 'rowwise':
      for d, alpha in zip(direction, alphas):
          for r in range(d.shape[0]):
              d[r,:].mul_(alpha[r,:].norm()/(d[r,:].norm() + 1e-10))
  elif norm == 'cellwise':
      # Rescale the entries in the direction so that each cell direction
      # has the unit norm.
      for d, alpha in zip(direction, alphas):
          d.mul_(alpha.norm()/(d.norm() + 1e-10))
  elif norm == 'modelwise':
      # Rescale the entries in the direction so that the model direction has
      # the unit norm.
      norm_d = 0.
      norm_a = 0.
      for d, a in zip(direction, alphas): 
        norm_d += (d*d).sum()
        norm_a += (a*a).sum()
      norm_d.sqrt_()
      norm_a.sqrt_()
      for d in direction:
        d.mul_(alpha/(norm+1e-10))
  else:
      raise(ValueError("Not implemented norm named as %s"%norm))
  return direction

def obtain_direction(model, valid_queue, direction_file=None, method='random', norm_type='cellwise'):
  if method == 'fromfile' and not os.path.exists(direction_file):
      raise(ValueError("For fromfile method, you should give a direction file name"))
  if method == 'fromfile' and not os.path.exists(direction_file):
      print("Since no direction file, we use default random method, and then save the direction to the file: %s"%direction_file)
      method = 'random'

  if method == 'fromfile':
    print("file exists, loading direction file from %s"%direction_file)
    d_dict = json.load(open(direction_file,'r'))
    d1 = d_dict['d1']
    d1 = [torch.from_numpy(np.array(d)) for d in d1]
    d2 = d_dict['d2']
    d2 = [torch.from_numpy(np.array(d)) for d in d2]

  elif method == 'random':
      print("direction as random Gaussian vector")
      d1 = create_random_direction(model.arch_parameters())    
      d2 = create_random_direction(model.arch_parameters())    

  elif method == 'grad':
      print("direction as the gradient of alpha")
      d1 = obtain_grad_direction(model, valid_queue)
      d2 = create_random_direction(model.arch_parameters())    

  else:
      raise(ValueError("Not implemented method named as %s"%method))

  # to get orthogonal direction
  norm = 0.
  inner_product = 0.
  for i, d in enumerate(d1): 
      norm += (d*d).sum()
      inner_product += (d*d2[i]).sum()
  norm.sqrt_()
  for i, d in enumerate(d1): d2[i] = d2[i] - d*inner_product/norm


  # norm
  alphas = [alpha.data.cpu() for alpha in model.arch_parameters()]
  d1 = norm_direction(d1, norm=norm_type, alphas=alphas)
  d2 = norm_direction(d2, norm=norm_type, alphas=alphas)

  # save direction to json file
  if not os.path.exists(direction_file):
    d_dict = {'d1': [d.tolist() for d in d1], 'd2':[d.tolist() for d in d2]}
    with open(direction_file, 'w') as f:
      json.dump(d_dict, f)
  return d1, d2

def infer(valid_queue, model, criterion, verbose=True):
  objs = utils.AverageMeter()
  top1 = utils.AverageMeter()
  top5 = utils.AverageMeter()
  model.eval()

  with torch.no_grad():
    for step, (input, target) in enumerate(valid_queue):
      input = Variable(input) #, volatile=True
      target = Variable(target) #, volatile=True
      if not args.disable_cuda:
        input = input.cuda()
        target = target.cuda(async=True)
  
      logits = model(input, 0)
      loss = criterion(logits, target)
  
      prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
      n = input.size(0)
      objs.update(loss.item(), n)
      top1.update(prec1.item(), n)
      top5.update(prec5.item(), n)
  
      if verbose and step % args.report_freq == 0:
        print('valid %03d %.3e %.3f %.3f'%(step, objs.avg, top1.avg, top5.avg))
      # if step > 10: break

  return top1.avg, objs.avg


def compute_landscape(base_dir, primitives):
  checkpoint_path = 'checkpoint_%d_%d.pth.tar' % (args.task_id, args.checkpoint_epoch-1)
  weight_file = os.path.join(base_dir, checkpoint_path)
#  alpha_file = os.path.join(base_dir, 'results_of_7q/alpha/49.txt')
#  alpha_value = load_alpha(alpha_file)
  beta_decay_scheduler.step(args.checkpoint_epoch-1)
  ckpt = torch.load(weight_file, map_location=lambda storage, loc: storage)
#  print(ckpt.keys())
  state_dict = ckpt['state_dict']
  alpha_value = [ckpt['alphas_normal'], ckpt['alphas_reduce']]
  del ckpt

  # construct model & load weights and alpha
  print(">>> Constructing model & load weights and alphas")
  criterion = nn.CrossEntropyLoss()
  if not args.disable_cuda:
    criterion = criterion.cuda()

  model = Network(args.init_channels, args.n_classes, args.layers, criterion,
                       primitives, steps=args.nodes, args=args)
  if not args.disable_cuda:
    model = model.cuda()

  model.load_state_dict(state_dict, strict=False)
  arch_params = model.arch_parameters()
  for i, alpha in enumerate(arch_params):
    alpha.data.copy_(alpha_value[i])

  # get data_loader
  print(">>> Constructing dataloader")
  train_transform, valid_transform = utils._data_transforms_cifar10(args)
  train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)

  num_train = len(train_data)
  indices = list(range(num_train))
  split = int(np.floor(args.train_portion * num_train))

  train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
      pin_memory=True, num_workers=2)

  valid_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
      pin_memory=True, num_workers=2)

  if args.test_infer:
    print(">>> First test whether the weight and alpha are loaded correctly")
    acc, loss = infer(valid_queue, model, criterion)
    print(acc, loss)

  landscape_dir = os.path.join(base_dir, 'vis_landscape')
  if not os.path.exists(landscape_dir):
    os.makedirs(landscape_dir)
  # obtain directions of pertubation
  print(">>> Obtaining direction of pertubation")
  direction_file = os.path.join(landscape_dir, 'direction-%d.json'%args.task_id)
  d1, d2 = obtain_direction(model, valid_queue, direction_file=direction_file, method='random', norm_type='cellwise')

  # compute landscape
  print(">>> Computing landscape")
  x = np.linspace(args.xmin, args.xmax, num=args.xnum)
  y = np.linspace(args.xmin, args.xmax, num=args.xnum)
  losses = np.zeros([len(x), len(y)])
  accs = np.zeros([len(x), len(y)])
  for i, delta1 in enumerate(x):
    for j, delta2 in enumerate(y):
        arch_params[0].data.copy_(alpha_value[0] + d1[0]*delta1+d2[0]*delta2)
        arch_params[1].data.copy_(alpha_value[1] + d1[1]*delta1+d2[1]*delta2)
        acc, loss = infer(valid_queue, model, criterion)
        losses[i, j] = loss
        accs[i, j] = acc
        print('x,y/acc/loss: %f,%f/%f/%f'%(delta1, delta2, acc, loss))

  # save loss & acc 
  results_file = os.path.join(landscape_dir, 'results-%d.csv'%args.task_id)
  title = ['X', 'Y'] + ['loss_%d'%i for i in range(losses.shape[1])] + ['acc_%d'%i for i in range(accs.shape[1])]
  with open(results_file, 'w') as f:
    writer = csv.writer(f)
    writer.writerow(title)
    for i in range(losses.shape[0]):
        row = [x[i], y[i]] + losses[i,:].tolist() + accs[i,:].tolist()
        writer.writerow(row)
  
  return x, y, losses, accs

def load_landscape(results_file):
  print(">>> Loading landscape from file %s"%results_file)
  with open(results_file) as f:
    reader = csv.reader(f)
    for i, row in enumerate(reader):
      if i == 0: 
         N = (len(row) - 2) / 2
         N = int(N)
         x = np.zeros([N])
         y = np.zeros([N])
         losses = np.zeros([N, N])
         accs = np.zeros([N, N])
         continue # jump the title
      x[i-1] = row[0]
      y[i-1] = row[1]
      losses[i-1,:] = row[2:N+2]
      accs[i-1,:] = row[N+2:]
  return x, y, losses, accs


if __name__ == '__main__':

  # base_dir = 'experiments/search_logs/%s/%s/'%(args.space, args.dataset)
  base_dir = args.save
  landscape_dir = os.path.join(base_dir, 'vis_landscape')
  results_file = os.path.join(landscape_dir, 'results-%d.csv'%args.task_id)

  if not args.disable_cuda:
    torch.cuda.set_device(args.gpu)
    print('set gpu device = %d' % args.gpu)

  if os.path.exists(results_file):
    x, y, losses, accs = load_landscape(results_file)
  else:
    space = spaces_dict[args.space]
    x, y, losses, accs = compute_landscape(base_dir, space)

  # plot loss landscape
  print(">>> Plotting landscape")
  X, Y = np.meshgrid(x, y)
  Z = losses
  vmin, vmax, vlevel = 0.0, 1.2, 0.03
  levels = np.arange(vmin, vmax, vlevel)
#  levels = np.concatenate((np.arange(0.2, 0.5, 0.1), np.arange(0.5, 4.0, 0.5), np.arange(4.0, 10, 2)), axis=-1)
  plot_2D(X, Y, Z, landscape_dir, 'valid_loss-%d'%args.task_id, levels, args.show, args.azim, args.elev)

  # plot acc landscape
  print(">>> Plotting landscape")
  X, Y = np.meshgrid(x, y)
  Z = accs / 100.
  vmin, vmax, vlevel = 0.4, 1.0, 0.01
  levels = np.arange(vmin, vmax, vlevel)
#  levels = np.concatenate((np.arange(0.1, 0.5, 0.1), np.arange(0.5, 0.7, 0.05), np.arange(0.7, 1.0, 0.02)), axis=-1)
  plot_2D(X, Y, Z, landscape_dir, 'valid_acc-%d'%args.task_id, levels, args.show, args.azim, args.elev)
