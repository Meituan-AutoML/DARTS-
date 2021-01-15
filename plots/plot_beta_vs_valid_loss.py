import sys
import matplotlib.pyplot as plt
import argparse
from parse import parse
import json

def read_file(file):
    accs = []
    betas = []
    with open(file, 'r') as f:
        for line in f.readlines():
            line = line.strip()

            res=parse("{} {} {} valid_acc {acc:f}", line)
            if res is not None:
                accs.append(res['acc'])

            res=parse("{} {} {} EPOCH {} SKIP BETA DECAY RATE: {beta:e}", line)
            if res is not None:
                betas.append(res['beta'])

    return accs,betas

def read_json(file):
    with open(file, 'r') as f:
        for line in f.readlines():
            json_dict = json.loads(line.strip())
    return json_dict

def gen_beta(beta_decay, beta=1.0):
    if beta_decay == 'linear':
        betas=range(50,0,-1)
        betas=[b*beta/50. for b in betas]
        return betas
    elif beta_decay == 'cosine':
        return None
    elif beta_decay == 'none': 
        return [0]*50


def plot(**kwargs):
    for key, val in kwargs.items():
        plt.plot(val, label=key)
    plt.legend()
    plt.tight_layout()
    # plt.savefig('train_loss_vs_beta.pdf')
    plt.show()
    

def plot_beta_acc(accs, betas):
    plt.plot(accs)
    plt.plot(betas)
    plt.show()

def plot_loss_beta(train_losses, val_losses, betas):
    color = ['orangered', 'blue', 'lightseagreen']
    _, ax = plt.subplots(figsize=(3,3))
    
    lns1 = ax.plot(train_losses, label='train loss', color=color[0])
    lns2 = ax.plot(val_losses, label='val loss', color=color[1])

    ax.set_xlabel('epoch')
    ax.set_ylabel(r'$L$', rotation=0)

    ax2 = ax.twinx()
    lns3 = ax2.plot(betas, label='beta', color=color[2])
    ax2.set_ylim(0,1)
    ax2.set_ylabel(r'$\beta$', rotation=0)

    # combine labels
    lns = lns1+lns2+lns3
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc=0)

    plt.title(r'$\beta_0$=%.1f'%betas[0])
    plt.tight_layout()
    plt.savefig('beta-%.1f-loss.pdf'%(betas[0]))
    plt.show()

def plot_loss_vs_beta(dict_set):
    _, axes = plt.subplots(1,2,figsize=(6,3))

    beta=[1,0.7,0.4,0.1,0]

    for d,b in zip(dict_set,beta):
        axes[0].plot(d['train_loss'], label=r'$\beta$=%.1f'%b)
        axes[1].plot(d['valid_loss'], label=r'$\beta$=%.1f'%b)
    
    axes[0].set_xlabel('epoch')
    axes[1].set_xlabel('epoch')
    axes[0].set_ylabel(r'$L_{train}$', rotation=0, labelpad=10)
    axes[1].set_ylabel(r'$L_{valid}$', rotation=0, labelpad=10)

    axes[0].set_title('CIFAR-10, S3')
    axes[1].set_title('CIFAR-10, S3')
    
    axes[0].legend()
    axes[1].legend()
    plt.tight_layout()
    plt.savefig('train-val-loss-vs-beta.pdf')
    plt.show()

def plot_train_loss_vs_beta(dict_set):
    _, axes = plt.subplots(figsize=(3,3))

    beta=[1,0.7,0.4,0.1,0]

    for d,b in zip(dict_set,beta):
        axes.plot(d['train_loss'], label=r'$\beta$=%.1f'%b)
    
    axes.set_xlabel('epoch')
    axes.set_ylabel(r'$L_{train}$', rotation=0, labelpad=10)

    axes.set_title('CIFAR-10, S3')
    axes.legend()
    plt.tight_layout()
    plt.savefig('train-loss-vs-beta.pdf')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("plot beta vs. val acc")
    # Example usage: 
    # python plot_beta_vs_valid_loss.py --json-dict experiments/search_logs/s3/cifar10/0.0_0.0003-0/errors_12.json --beta-decay linear --beta 0.4
    # python plot_beta_vs_valid_loss.py --json-dict ../experiments/search_logs/s3/cifar10/0.0_0.0003-0/errors_16.json ../experiments/search_logs/s3/cifar10/0.0_0.0003-0/errors_0.json ../experiments/search_logs/s3/cifar10/0.0_0.0003-0/errors_10.json ../experiments/search_logs/s3/cifar10/0.0_0.0003-0/errors_4.json ../experiments/search_logs/s3/cifar10/0.0_0.0003-0/errors_13.json --beta 1.0 0.7 0.4 0.1 0
    parser.add_argument('--log', type=str, default='train-search-darts-s5-cifar10-job-0-task-33-drop-0.0-seed-33.log', help='add an auxiliary skip')
    parser.add_argument('--json-dict', type=str, default='../experiments/search_logs/s5/cifar10/0.0_0.0003-0/errors_33.json', nargs="+", help='error dict json file')
    parser.add_argument('--beta-decay', default='linear', choices=['linear','cosine', 'none'], help='type of beta decay')
    parser.add_argument('--beta', default=1.0, type=float, nargs='+', help='initial beta')
    args = parser.parse_args()
    # accs, betas = read_file(args.log)

    plt.rcParams["font.family"] = "Times New Roman"
    # plot_beta_acc(accs, betas)
    
    # plot(train_loss=json_dict['train_loss'], val_loss=json_dict['valid_loss'], beta=betas)

    dict_set = []
    for json_d, beta in zip(args.json_dict, args.beta):
        json_dict = read_json(json_d)
        betas = gen_beta(args.beta_decay, beta)
        dict_set.append(json_dict)
        # plot_loss_beta(json_dict['train_loss'], json_dict['valid_loss'], betas)

    # plot_loss_vs_beta(dict_set)
    plot_train_loss_vs_beta(dict_set)

