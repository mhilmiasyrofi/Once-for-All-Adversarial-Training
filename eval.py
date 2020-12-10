import os, argparse, time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.optim import SGD, Adam, lr_scheduler

from models.cifar10.resnet18_OAT import ResNet18OAT
# from models.cifar10.resnet_OAT import ResNet34OAT
# from models.svhn.wide_resnet_OAT import WRN16_8OAT
# from models.stl10.wide_resnet_OAT import WRN40_2OAT

from dataloaders.cifar10 import cifar10_dataloaders, get_adversarial_images
from dataloaders.svhn import svhn_dataloaders
from dataloaders.stl10 import stl10_dataloaders

from utils.utils import *
from utils.context import ctx_noparamgrad_and_eval
from utils.sample_lambda import element_wise_sample_lambda, batch_wise_sample_lambda

parser = argparse.ArgumentParser(description='cifar10 Training')
parser.add_argument('--gpu', default='7')
parser.add_argument('--cpus', default=4)
# dataset:
parser.add_argument('--dataset', '--ds', default='cifar10', choices=['cifar10', 'svhn', 'stl10'], help='which dataset to use')
parser.add_argument('--train_adversarial', '--tr', default='autoattack', help='which adversarial data used to train')
parser.add_argument('--test_adversarial', '--ts', default='cw', help='which adversarial data used to test')
# optimization parameters:
parser.add_argument('--batch_size', '-b', default=128, type=int, help='mini-batch size')
parser.add_argument('--epochs', '-e', default=200, type=int, help='number of total epochs to run')
parser.add_argument('--decay_epochs', '--de', default=[50,150], nargs='+', type=int, help='milestones for multisteps lr decay')
parser.add_argument('--opt', default='sgd', choices=['sgd', 'adam'], help='which optimizer to use')
parser.add_argument('--decay', default='cos', choices=['cos', 'multisteps'], help='which lr decay method to use')
parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--wd', default=5e-4, type=float, help='weight decay')
# adv parameters:
parser.add_argument('--targeted', action='store_true', help='If true, targeted attack')
parser.add_argument('--eps', type=int, default=8)
parser.add_argument('--steps', type=int, default=7)
# OAT parameters:
parser.add_argument('--distribution', default='disc', choices=['disc'], help='Lambda distribution')
parser.add_argument('--lambda_choices', default=[0.0,0.1,0.2,0.3,0.4,1.0], nargs='*', type=float, help='possible lambda values to sample during training')
parser.add_argument('--probs', default=-1, type=float, help='the probs for sample 0, if not uniform sample')
parser.add_argument('--encoding', default='rand', choices=['none', 'onehot', 'dct', 'rand'], help='encoding scheme for Lambda')
parser.add_argument('--dim', default=128, type=int, help='encoding dimention for Lambda')
parser.add_argument('--use2BN', action='store_true', help='If true, use dual BN')
parser.add_argument('--sampling', default='ew', choices=['ew', 'bw'], help='sampling scheme for Lambda')
# others:
parser.add_argument('--resume', action='store_true', help='If true, resume from early stopped ckpt')
args = parser.parse_args()
args.efficient = True
print(args)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
torch.backends.cudnn.benchmark = True

# data loader:
if args.dataset == 'cifar10':
    train_loader, test_loader = cifar10_dataloaders(train_batch_size=args.batch_size, num_workers=args.cpus)
    _, test_adv_loader = get_adversarial_images(adversarial_data=args.test_adversarial, batch_size=args.batch_size)
elif args.dataset == 'svhn':
    train_loader, val_loader, _ = svhn_dataloaders(train_batch_size=args.batch_size, num_workers=args.cpus)
elif args.dataset == 'stl10':
    train_loader, val_loader = stl10_dataloaders(train_batch_size=args.batch_size, num_workers=args.cpus)

print("Train: ", args.train_adversarial)
print("Test: ", args.test_adversarial)

# model:
if args.encoding in ['onehot', 'dct', 'rand']:
    FiLM_in_channels = args.dim
else: # non encoding
    FiLM_in_channels = 1
if args.dataset == 'cifar10':
    model_fn = ResNet18OAT
# elif args.dataset == 'svhn':
#     model_fn = WRN16_8OAT
# elif args.dataset == 'stl10':
#     model_fn = WRN40_2OAT
model = model_fn(use2BN=args.use2BN, FiLM_in_channels=FiLM_in_channels).cuda()
model = torch.nn.DataParallel(model)
# for name, p in model.named_parameters():
#     print(name, p.size())

# mkdirs:
model_str = os.path.join(model_fn.__name__)
if args.use2BN:
    model_str += '-2BN'
if args.opt == 'sgd':
    opt_str = 'e%d-b%d_sgd-lr%s-m%s-wd%s' % (args.epochs, args.batch_size, args.lr, args.momentum, args.wd)
elif args.opt == 'adam':
    opt_str = 'e%d-b%d_adam-lr%s-wd%s' % (args.epochs, args.batch_size, args.lr, args.wd)
if args.decay == 'cos':
    decay_str = 'cos'
elif args.decay == 'multisteps':
    decay_str = 'multisteps-%s' % args.decay_epochs
attack_str = 'targeted' if args.targeted else 'untargeted' + '-pgd-%s-%d' % (args.eps, args.steps)
lambda_str = '%s-%s-%s' % (args.distribution, args.sampling, args.lambda_choices)
if args.probs > 0:
    lambda_str += '-%s' % args.probs
if args.encoding in ['onehot', 'dct', 'rand']:
    lambda_str += '-%s-d%s' % (args.encoding, args.dim)
save_folder = os.path.join('results', 'cifar10', model_str, '%s_%s_%s_%s' % (args.train_adversarial, opt_str, decay_str, lambda_str))


print("save folder: ")
print(save_folder)
if args.train_adversarial == "original" :
    create_dir(save_folder)

# encoding matrix:
if args.encoding == 'onehot':
    I_mat = np.eye(args.dim)
    encoding_mat = I_mat
elif args.encoding == 'dct':
    from scipy.fftpack import dct
    dct_mat = dct(np.eye(args.dim), axis=0)
    encoding_mat = dct_mat
elif args.encoding == 'rand':
    rand_mat = np.random.randn(args.dim, args.dim)
    np.save(os.path.join(save_folder, 'rand_mat.npy'), rand_mat)
    rand_otho_mat, _ = np.linalg.qr(rand_mat)
    np.save(os.path.join(save_folder, 'rand_otho_mat.npy'), rand_otho_mat)
    encoding_mat = rand_otho_mat
elif args.encoding == 'none':
    encoding_mat = None

# val_lambdas:
if args.distribution == 'disc':
    val_lambdas = args.lambda_choices
else:
    val_lambdas = [0,0.2,0.5,1]

# optimizer:
if args.opt == 'sgd':
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
elif args.opt == 'adam':
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
if args.decay == 'cos':
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
elif args.decay == 'multisteps':
    scheduler = lr_scheduler.MultiStepLR(optimizer, args.decay_epochs, gamma=0.1)

if args.train_adversarial == "original" :
    # load pretrained model
    pretrained_path = "results/cifar10/ResNet18OAT-2BN/original_e50-b100_sgd-lr0.1-m0.9-wd0.0005_cos_disc-ew-[0.0, 0.1, 0.2, 0.3, 0.4, 1.0]-rand-d128/best_TA1.0.pth"
    state_dict = torch.load(pretrained_path)
    model.load_state_dict(state_dict)
else :
    # load adversarially trained model
    last_epoch, best_TA, best_ATA, training_loss, val_TA, val_ATA \
         = load_ckpt(model, optimizer, scheduler, os.path.join(save_folder, 'latest.pth'))
    start_epoch = last_epoch + 1
    


# attacker:
# attacker = PGD(eps=args.eps/255, steps=args.steps, use_FiLM=True)
attacker = None

eval_folder = os.path.join(save_folder, 'eval/')
create_dir(eval_folder)


val_fp = open(os.path.join(eval_folder, args.test_adversarial + '.txt'), 'a+')
start_time = time.time()

## validation:
model.eval()
requires_grad_(model, False)

val_accs, val_accs_adv = {}, {}
for val_lambda in val_lambdas:
    val_accs[val_lambda], val_accs_adv[val_lambda] = AverageMeter(), AverageMeter()
    
val_lambdas = [1.]

for i, (batch, adv_batch) in enumerate(zip(test_loader,test_adv_loader)):
    imgs = batch["input"]
    labels = batch["target"]

    adv_imgs = adv_batch["input"]
    adv_labels = adv_batch["target"]

    for j, val_lambda in enumerate(val_lambdas):
        # sample _lambda:
        if args.distribution == 'disc' and encoding_mat is not None:
            _lambda = np.expand_dims( np.repeat(j, labels.size()[0]), axis=1 ).astype(np.uint8)
            _lambda = encoding_mat[_lambda,:] 
        else:
            _lambda = np.expand_dims( np.repeat(val_lambda, labels.size()[0]), axis=1 )
        _lambda = torch.from_numpy(_lambda).float().cuda()
        if args.use2BN:
            idx2BN = int(labels.size()[0]) if val_lambda==0 else 0
        else:
            idx2BN = None
        # TA:
        logits = model(imgs, _lambda, idx2BN)
        val_accs[val_lambda].append((logits.argmax(1) == labels).float().mean().item())

        logits_adv = model(adv_imgs, _lambda, idx2BN)
        val_accs_adv[val_lambda].append((logits_adv.argmax(1) == adv_labels).float().mean().item())

val_str = 'Validation | Time: %.4f\n' % ((time.time()-start_time))
for val_lambda in val_lambdas:
    val_str += 'val_lambda%s: TA: %.4f, ATA: %.4f\n' % (val_lambda, val_accs[val_lambda].avg, val_accs_adv[val_lambda].avg)
print(val_str)
val_fp.write(val_str + '\n')
val_fp.close() # close file pointer