import argparse
import os
import sys
import time

import numpy as np
import torch

parser = argparse.ArgumentParser(description='baseline')
parser.add_argument('--testname', type=str, default=None)
parser.add_argument('--debug', action='store_true')
parser.add_argument('--dataset', type=str, default=None)
parser.add_argument('--regime', type=str, default=None)
parser.add_argument('--path', type=str)
parser.add_argument('--subset', type=str, default=None)
parser.add_argument('--ratio', type=float, default=None)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--img_size', type=int, default=80)

parser.add_argument('--seed', type=int, default=12345)
parser.add_argument('--model_name', type=str, default='mrnet')
parser.add_argument('--r_func', type=str, default='dist')
parser.add_argument('--contrast', dest='contrast', action='store_true')
parser.add_argument('--levels', type=str, default='111')
parser.add_argument('--dropout', dest='dropout', action='store_true')
parser.add_argument('--no_rc', dest='row_col', action='store_false')
parser.add_argument('--relu_bf', dest='relu_before_reduce', action='store_true')

parser.add_argument('--epochs', type=int, default=-1)
parser.add_argument('--device', type=int, default=0)

parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--beta1', type=float, default=0.9)
parser.add_argument('--beta2', type=float, default=0.999)
parser.add_argument('--epsilon', type=float, default=1e-8)
parser.add_argument('--wd', type=float, default=0)
parser.add_argument('--use_tag', type=int, default=1)
parser.add_argument('--meta_beta', type=float, default=0.)
parser.add_argument('--early_stopping', type=int, default=20)
parser.add_argument('--loss_func', type=str, default='contrast')
parser.add_argument('--w_loss', dest='weighted_loss', action='store_true')
parser.add_argument('--flip', dest='flip', action='store_true')
parser.add_argument('--force_bias', dest='force_bias', action='store_true')

parser.add_argument('--recovery', action='store_true')
parser.add_argument('--test', action='store_true')
parser.add_argument('--multihead', action='store_true')
parser.add_argument('--multihead_mode', type=str, default=None)
parser.add_argument('--multihead_w', type=float, default=1.)


def check_paths(args):
    try:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)
        new_log_dir = os.path.join(args.log_dir, time.ctime().replace(" ", "-"))
        args.log_dir = new_log_dir
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)
        if not os.path.exists(args.checkpoint_dir):
            os.makedirs(args.checkpoint_dir)
    except OSError as e:
        print(e)
        sys.exit(1)


def main(args):
    args.cuda = torch.cuda.is_available()

    if args.test:
        assert args.recovery

    if not args.recovery:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.cuda:
            torch.cuda.manual_seed(args.seed)

    from trainer import Trainer
    trainer = Trainer(args)
    if not args.test:
        trainer.main()
    else:
        for subset in ['val', 'test']:
            loss, acc, acc_regime = trainer.evaluate(subset)
            print(f'{subset} loss: {loss} - accuracy: {acc}')
            if acc_regime is not None:
                print(f'{subset} In Regime:')
                for key, val in acc_regime.items():
                    if isinstance(val, float):
                        print(f'{key}: {val:.3f}')
                    else:
                        print(f'{key}: {val}')


if __name__ == "__main__":
    args = parser.parse_args()

    if args.testname is None:
        args.testname = args.dataset

    main(args)
    print(f"script: {sys.argv[0]}")
    print(f"args: {str(sys.argv)}")
