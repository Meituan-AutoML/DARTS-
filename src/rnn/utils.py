import torch
import torch.nn as nn
import torch.nn.functional as F
import os, shutil
import numpy as np
from torch.autograd import Variable
import math

def repackage_hidden(h):
    #print("calling __func__")
    if type(h) in [Variable, torch.Tensor]:
        #print("shape of hidden:", h.size())
        return Variable(h.data)
    else:
        # list
        return tuple(repackage_hidden(v) for v in h)


def batchify(data, bsz, args):
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.view(bsz, -1).t().contiguous()
    print(data.size())
    if args.cuda:
        data = data.cuda()
    return data


def get_batch(source, i, args, seq_len=None, evaluation=False):
    seq_len = min(seq_len if seq_len else args.bptt, len(source) - 1 - i)
    data = Variable(source[i:i+seq_len], volatile=evaluation)
    target = Variable(source[i+1:i+1+seq_len])
    return data, target


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)

    print('Experiment dir : {}'.format(path))
    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)


def save_checkpoint(model, optimizer, epoch, path, finetune=False):
    if finetune:
        torch.save(model, os.path.join(path, 'finetune_model.pt'))
        torch.save(optimizer.state_dict(), os.path.join(path, 'finetune_optimizer.pt'))
    else:
        torch.save(model, os.path.join(path, 'model.pt'))
        torch.save(optimizer.state_dict(), os.path.join(path, 'optimizer.pt'))
    torch.save({'epoch': epoch+1}, os.path.join(path, 'misc.pt'))


def embedded_dropout(embed, words, dropout=0.1, scale=None):
    if dropout:
        mask = embed.weight.data.new().resize_((embed.weight.size(0), 1)).bernoulli_(1 - dropout).expand_as(embed.weight) / (1 - dropout)
        mask = Variable(mask)
        masked_embed_weight = mask * embed.weight
    else:
        masked_embed_weight = embed.weight
    if scale:
        masked_embed_weight = scale.expand_as(masked_embed_weight) * masked_embed_weight

    padding_idx = embed.padding_idx
    if padding_idx is None:
        padding_idx = -1

    #X = embed._backend.Embedding.apply(words, masked_embed_weight,
    X = F.embedding(words, masked_embed_weight,
        padding_idx, embed.max_norm, embed.norm_type,
        embed.scale_grad_by_freq, embed.sparse
    )
    return X


class LockedDropout(nn.Module):
    def __init__(self):
        super(LockedDropout, self).__init__()

    def forward(self, x, dropout=0.5):
        if not self.training or not dropout:
            return x
        m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - dropout)
        mask = Variable(m.div_(1 - dropout), requires_grad=False)
        mask = mask.expand_as(x)
        return mask * x


def mask2d(B, D, keep_prob, cuda=True):
    m = torch.floor(torch.rand(B, D) + keep_prob) / keep_prob
    m = Variable(m, requires_grad=False)
    if cuda:
        m = m.cuda()
    return m

class DecayScheduler(object):
    def __init__(self, base_lr=1.0, last_iter=-1, T_max=50, T_stop=50, decay_type='cosine'):
        self.base_lr = base_lr
        self.T_max = T_max
        self.T_stop = T_stop
        self.cnt = 0
        self.decay_type = decay_type
        self.decay_rate = 1.0

    def step(self, epoch):
        if self.decay_type == "cosine":
            self.decay_rate = self.base_lr * (1 + math.cos(math.pi * epoch / self.T_max)) / 2.0 if epoch <= self.T_stop else self.decay_rate
        elif self.decay_type == "slow_cosine":
            self.decay_rate = self.base_lr * math.cos((math.pi/2) * epoch / self.T_max) if epoch <= self.T_stop else self.decay_rate
        elif self.decay_type == "linear":
            self.decay_rate = self.base_lr * (self.T_max - epoch) / self.T_max if epoch <= self.T_stop else self.decay_rate
        else:
            self.decay_rate = self.base_lr