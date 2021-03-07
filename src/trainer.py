import os
import pickle

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

import criteria
from report_acc_regime import init_acc_regime, update_acc_regime

torch.backends.cudnn.benchmark = True


def renormalize(images):
    return (images / 255 - 0.5) * 2


class Trainer:
    def __init__(self, args):
        self.args = args
        self.args.cuda = torch.cuda.is_available()
        torch.cuda.set_device(self.args.device)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        if self.args.cuda:
            torch.cuda.manual_seed(self.args.seed)

        test_path = os.path.join('.', 'results', self.args.testname)
        self.save_path = os.path.join(test_path, 'save')
        self.log_path = os.path.join(test_path, 'log')

        if args.recovery:
            assert os.path.isdir(test_path), 'Recovery directory (' + test_path + ') does not exist'
            print('Recovering an existing run')
        else:
            if os.path.isdir(test_path):
                print(f'Removing existing save directory at {test_path}')
                import shutil
                shutil.rmtree(test_path)

            print(f'Creating new save directory at {test_path}')
            os.makedirs(self.save_path)
            os.makedirs(self.log_path)

            import yaml
            cfg_file = os.path.join(self.save_path, 'cfg.yml')
            with open(cfg_file, 'w') as f:
                yaml.dump(args.__dict__, f, default_flow_style=False)

        print('Loading datasets')
        from data.data_utils import get_data
        self.trainloader = get_data(self.args.path, self.args.dataset, self.args.img_size,
                                    dataset_type="train", regime=self.args.regime, subset=self.args.subset,
                                    batch_size=self.args.batch_size, drop_last=True, num_workers=self.args.num_workers,
                                    ratio=self.args.ratio, shuffle=True)
        self.validloader = get_data(self.args.path, self.args.dataset, self.args.img_size,
                                    dataset_type="val", regime=self.args.regime, subset=self.args.subset,
                                    batch_size=self.args.batch_size, drop_last=True, num_workers=self.args.num_workers,
                                    ratio=self.args.ratio, shuffle=False)
        self.testloader = get_data(self.args.path, self.args.dataset, self.args.img_size,
                                   dataset_type="test", regime=self.args.regime, subset=self.args.subset,
                                   batch_size=self.args.batch_size, drop_last=True, num_workers=self.args.num_workers,
                                   ratio=self.args.ratio, shuffle=False)

        print('Building model')
        params = args.__dict__
        print(args)
        params['num_meta'] = 9 if 'RAVEN' in self.args.dataset else 12
        self.use_meta = 0 if args.meta_beta == 0 else params['num_meta']

        assert args.model_name == 'mrnet'
        from networks.mrnet import MRNet
        self.model = MRNet(use_meta=self.use_meta, dropout=args.dropout, force_bias=args.force_bias,
                           reduce_func=args.r_func, levels=args.levels, do_contrast=args.contrast,
                           multihead=args.multihead)

        if self.args.cuda:
            self.model.cuda()

        self.optimizer = optim.Adam([param for param in self.model.parameters() if param.requires_grad],
                                    self.args.lr, betas=(self.args.beta1, self.args.beta2), eps=self.args.epsilon,
                                    weight_decay=self.args.wd)

        if args.recovery:
            ckpt_file_path = os.path.join(self.save_path, 'model.pth')
            print(f'Loading existing checkpoint from {ckpt_file_path}')
            state_dict = self.model.state_dict()
            new_state_dict = torch.load(ckpt_file_path, map_location=lambda storage, loc: storage)
            for key, val in new_state_dict.items():
                state_dict[key] = val
            self.model.load_state_dict(state_dict)

        if self.args.loss_func == 'contrast':
            self.criterion = lambda x, y, reduction='mean': criteria.contrast_loss(x, y, reduction, args.weighted_loss)
        elif self.args.loss_func == 'ce':
            import torch.nn as nn
            self.criterion = nn.CrossEntropyLoss()
        if self.use_meta:
            self.criterion_meta = criteria.type_loss

    def train(self, epoch):
        self.model.train()

        counter = 0
        loss_avg = 0.0
        loss_meta_avg = 0.0
        acc_avg = 0.0
        acc_multihead_avg = [0.0] * 3

        for batch_data in tqdm(self.trainloader, f'Train epoch {epoch}'):
            counter += 1

            image, target, meta_target, structure_encoded, data_file = batch_data
            image = renormalize(image)

            image = image.cuda()
            target = target.cuda()
            if self.use_meta:
                meta_target = meta_target.cuda()

            model_outputs = self.model(image)
            if len(model_outputs) == 3:
                model_output, meta_pred, model_output_heads = model_outputs
            else:
                model_output, meta_pred = model_outputs
                model_output_heads = None

            loss = self.criterion(model_output, target)
            loss_avg += loss.item()
            acc = criteria.calculate_acc(model_output, target)
            acc_avg += acc.item()
            if self.use_meta:
                if len(meta_pred.shape) > 2:
                    meta_pred = meta_pred[range(meta_pred.shape[0]), target]
                loss_meta = self.criterion_meta(meta_pred, meta_target)
                loss_meta_avg += loss_meta.item()
                loss += self.use_meta * loss_meta
            if self.args.multihead:
                loss_head = [self.criterion(output, target, reduction='none') for output in model_output_heads]
                target_one_hot = torch.zeros_like(model_output_heads[0])
                target_one_hot.scatter_(1, target.view(-1, 1), 1.0)

                if self.args.multihead_mode is None:
                    weights = [1 / len(model_output_heads)] * len(model_output_heads)
                else:
                    probs = [target_one_hot * output.detach().sigmoid() + (1 - target_one_hot) * (1 - output.detach().sigmoid())
                             for output in model_output_heads]
                    if self.args.multihead_mode == 'prob':
                        probs_sum = sum(probs)
                        weights = [prob / probs_sum for prob in probs]
                    elif self.args.multihead_mode == 'eprob':
                        e_probs = [prob.exp() for prob in probs]
                        e_probs_sum = sum(e_probs)
                        weights = [prob / e_probs_sum for prob in e_probs]
                    else:
                        raise ValueError(f'Unsupported argument for multihead_mode: {self.args.multihead_mode}')

                loss_multihead = sum([weights[i] * loss_head[i] for i in range(len(loss_head))]).mean()
                loss += self.args.multihead_w * loss_multihead
                acc_multihead = [criteria.calculate_acc(x, target) for x in model_output_heads]
                for i, x in enumerate(acc_multihead):
                    acc_multihead_avg[i] += x.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.use_meta:
            print("Epoch {}, Train Avg Loss: {:.6f} META: {:.6f}, Train Avg Acc: {:.4f}".format(
                epoch, loss_avg / float(counter), loss_meta_avg / float(counter), acc_avg / float(counter)))
        else:
            print("Epoch {}, Train Avg Loss: {:.6f}, Train Avg Acc: {:.4f}".format(
                epoch, loss_avg / float(counter), acc_avg / float(counter)))
        if self.args.multihead:
            print(f'Multihead: {[x / float(counter) for x in acc_multihead_avg]}')

        return loss_avg / float(counter), acc_avg / float(counter)

    def validate(self, epoch):
        self.model.eval()

        counter = 0
        loss_avg = 0.0
        loss_meta_avg = 0.0
        acc_avg = 0.0
        acc_multihead_avg = [0.0] * 3

        acc_regime = init_acc_regime(self.args.dataset)

        for batch_data in tqdm(self.validloader, f'Valid epoch {epoch}'):
            counter += 1

            image, target, meta_target, structure_encoded, data_file = batch_data
            image = renormalize(image)

            image = image.cuda()
            target = target.cuda()
            if self.use_meta:
                meta_target = meta_target.cuda()

            with torch.no_grad():
                model_outputs = self.model(image)
                if len(model_outputs) == 3:
                    model_output, meta_pred, model_output_heads = model_outputs
                else:
                    model_output, meta_pred = model_outputs
                    model_output_heads = None

            loss = self.criterion(model_output, target)
            loss_avg += loss.item()
            acc = criteria.calculate_acc(model_output, target)
            acc_avg += acc.item()
            if self.use_meta:
                if len(meta_pred.shape) > 2:
                    meta_pred = meta_pred[range(meta_pred.shape[0]), target]
                loss_meta = self.criterion_meta(meta_pred, meta_target)
                loss_meta_avg += loss_meta.item()
            if self.args.multihead:
                acc_multihead = [criteria.calculate_acc(x, target) for x in model_output_heads]
                for i, x in enumerate(acc_multihead):
                    acc_multihead_avg[i] += x.item()

            if acc_regime is not None:
                update_acc_regime(self.args.dataset, acc_regime, model_output, target, structure_encoded, data_file)

        if counter > 0:
            if self.use_meta:
                print("Epoch {}, Valid Avg Loss: {:.6f} META {:.6f}, Valid Avg Acc: {:.4f}".format(
                    epoch, loss_avg / float(counter), loss_meta_avg / float(counter), acc_avg / float(counter)))
            else:
                print("Epoch {}, Valid Avg Loss: {:.6f}, Valid Avg Acc: {:.4f}".format(
                    epoch, loss_avg / float(counter), acc_avg / float(counter)))
            if self.args.multihead:
                print(f'Multihead: {[x / float(counter) for x in acc_multihead_avg]}')

        if acc_regime is not None:
            for key in acc_regime.keys():
                if acc_regime[key] is not None:
                    if acc_regime[key][1] > 0:
                        acc_regime[key] = float(acc_regime[key][0]) / acc_regime[key][1] * 100
                    else:
                        acc_regime[key] = None

        return loss_avg / float(counter), acc_avg / float(counter), acc_regime

    def test(self, epoch):
        self.model.eval()

        counter = 0
        loss_avg = 0.0
        loss_meta_avg = 0.0
        acc_avg = 0.0
        acc_multihead_avg = [0.0] * 3

        acc_regime = init_acc_regime(self.args.dataset)

        for batch_data in tqdm(self.testloader, f'Test epoch {epoch}'):
            counter += 1

            image, target, meta_target, structure_encoded, data_file = batch_data
            image = renormalize(image)

            image = image.cuda()
            target = target.cuda()
            if self.use_meta:
                meta_target = meta_target.cuda()

            with torch.no_grad():
                model_outputs = self.model(image)
                if len(model_outputs) == 3:
                    model_output, meta_pred, model_output_heads = model_outputs
                else:
                    model_output, meta_pred = model_outputs
                    model_output_heads = None

            loss = self.criterion(model_output, target)
            loss_avg += loss.item()
            acc = criteria.calculate_acc(model_output, target)
            acc_avg += acc.item()
            if self.use_meta:
                if len(meta_pred.shape) > 2:
                    meta_pred = meta_pred[range(meta_pred.shape[0]), target]
                loss_meta = self.criterion_meta(meta_pred, meta_target)
                loss_meta_avg += loss_meta.item()
            if self.args.multihead:
                acc_multihead = [criteria.calculate_acc(x, target) for x in model_output_heads]
                for i, x in enumerate(acc_multihead):
                    acc_multihead_avg[i] += x.item()

            if acc_regime is not None:
                update_acc_regime(self.args.dataset, acc_regime, model_output, target, structure_encoded, data_file)

        if counter > 0:
            if self.use_meta:
                print("Epoch {}, Test  Avg Loss: {:.6f} META {:.6f}, Test  Avg Acc: {:.4f}".format(
                    epoch, loss_avg / float(counter), loss_meta_avg / float(counter), acc_avg / float(counter)))
            else:
                print("Epoch {}, Test  Avg Loss: {:.6f}, Test  Avg Acc: {:.4f}".format(
                    epoch, loss_avg / float(counter), acc_avg / float(counter)))
            if self.args.multihead:
                print(f'Multihead: {[x / float(counter) for x in acc_multihead_avg]}')

        if acc_regime is not None:
            for key in acc_regime.keys():
                if acc_regime[key] is not None:
                    if acc_regime[key][1] > 0:
                        acc_regime[key] = float(acc_regime[key][0]) / acc_regime[key][1] * 100
                    else:
                        acc_regime[key] = None

        return loss_avg / float(counter), acc_avg / float(counter), acc_regime

    def evaluate(self, subset):
        self.model.eval()

        counter = 0
        loss_avg = 0.0
        loss_meta_avg = 0.0
        acc_avg = 0.0
        acc_multihead_avg = [0.0] * 3

        acc_regime = init_acc_regime(self.args.dataset)

        if subset == 'train':
            loader = self.trainloader
        elif subset == 'val':
            loader = self.validloader
        else:
            loader = self.testloader
        for batch_data in tqdm(loader, subset):
            counter += 1

            image, target, meta_target, structure_encoded, data_file = batch_data
            image = renormalize(image)

            image = image.cuda()
            target = target.cuda()
            if self.use_meta:
                meta_target = meta_target.cuda()

            with torch.no_grad():
                model_outputs = self.model(image)
                if len(model_outputs) == 3:
                    model_output, meta_pred, model_output_heads = model_outputs
                else:
                    model_output, meta_pred = model_outputs
                    model_output_heads = None

            loss = self.criterion(model_output, target)
            loss_avg += loss.item()
            acc = criteria.calculate_acc(model_output, target)
            acc_avg += acc.item()
            if self.use_meta:
                if len(meta_pred.shape) > 2:
                    meta_pred = meta_pred[range(meta_pred.shape[0]), target]
                loss_meta = self.criterion_meta(meta_pred, meta_target)
                loss_meta_avg += loss_meta.item()
            if self.args.multihead:
                acc_multihead = [criteria.calculate_acc(x, target) for x in model_output_heads]
                for i, x in enumerate(acc_multihead):
                    acc_multihead_avg[i] += x.item()

            if acc_regime is not None:
                update_acc_regime(self.args.dataset, acc_regime, model_output, target, structure_encoded, data_file)

        if counter > 0:
            if self.use_meta:
                print("{} - Avg Loss: {:.6f} META {:.6f}, Test  Avg Acc: {:.4f}".format(
                    subset, loss_avg / float(counter), loss_meta_avg / float(counter), acc_avg / float(counter)))
            else:
                print("{} - Avg Loss: {:.6f}, Test  Avg Acc: {:.4f}".format(
                    subset, loss_avg / float(counter), acc_avg / float(counter)))
            if self.args.multihead:
                print(f'Multihead: {[x / float(counter) for x in acc_multihead_avg]}')

        if acc_regime is not None:
            for key in acc_regime.keys():
                if acc_regime[key] is not None:
                    if acc_regime[key][1] > 0:
                        acc_regime[key] = float(acc_regime[key][0]) / acc_regime[key][1] * 100
                    else:
                        acc_regime[key] = None

        return loss_avg / float(counter), acc_avg / float(counter), acc_regime

    def main(self):
        self.t = []
        self.val_acc = []
        self.val_acc_regime = {}
        epoch_start = 0
        if self.args.recovery and os.path.isfile(os.path.join(self.save_path, 'peformance.pickle')):
            with open(os.path.join(self.save_path, 'peformance.pickle'), 'rb') as f:
                saved_acc = pickle.load(f)
            self.t = saved_acc['t']
            self.val_acc = saved_acc['accuracy']
            self.val_acc_regime = saved_acc['acc_regime']
            epoch_start = saved_acc['epoch']

        best_val_acc = 0
        best_val_acc_epoch = 0
        best_val_test_acc = 0
        best_test_acc = 0
        val_acc_regime = None
        print('-----')

        epoch = epoch_start
        while True:
            if self.args.epochs > 0 and epoch >= self.args.epochs:
                break
            epoch += 1

            train_loss, train_acc = self.train(epoch)
            val_loss, val_acc, val_acc_regime = self.validate(epoch)
            self.t.append(len(self.trainloader) * epoch)
            self.val_acc.append(val_acc)
            if val_acc_regime is not None:
                for key, val in val_acc_regime.items():
                    self.val_acc_regime.setdefault(key, []).append(val)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_acc_regime = val_acc_regime
                best_val_acc_epoch = epoch

                # Run test
                _, best_val_test_acc, test_acc_regime = self.test(epoch)

                if best_val_test_acc > best_test_acc:
                    best_test_acc = best_val_test_acc
                if val_acc_regime is not None:
                    print('Val In Regime:')
                    for key in val_acc_regime.keys():
                        if val_acc_regime[key] is not None:
                            print(f'{key}: {val_acc_regime[key]:.3f} / ', end='')
                        else:
                            print(f'{key}: {None} / ', end='')
                        if test_acc_regime[key] is not None:
                            print(f'{test_acc_regime[key]:.3f}')
                        else:
                            print(f'{test_acc_regime[key]:}')

                # Save model
                ckpt_file_path = os.path.join(self.save_path, 'model.pth')
                torch.save(self.model.state_dict(), ckpt_file_path)
                with open(os.path.join(self.save_path, 'peformance.pickle'), 'wb') as f:
                    pickle.dump({'epoch': epoch,
                                 't': self.t,
                                 'accuracy': self.val_acc,
                                 'acc_regime': self.val_acc_regime}, f)

            if self.args.early_stopping:
                if epoch - best_val_acc_epoch >= self.args.early_stopping:
                    print(f'Early stopping exit: {epoch - best_val_acc_epoch} > {self.args.early_stopping}')
                    break
                print(f"Early stopping countdown: {epoch - best_val_acc_epoch}/{self.args.early_stopping} (Best VAL: {best_val_acc:0.5f}, Best VAL TEST: {best_val_test_acc:0.5f}, Best TEST: {best_test_acc:0.5f})")

        print('Done Training')
        print(f'Best Validation Accuracy: {best_val_acc}')
        print(f'Best Validation Test Accuracy: {best_val_test_acc}')
        print(f'Best Test Accuracy: {best_test_acc}')

        if best_val_acc_regime is not None:
            print('Val In Regime:')
            for key in best_val_acc_regime.keys():
                if best_val_acc_regime[key] is not None:
                    print(f'{key}: {best_val_acc_regime[key]:.3f} / ', end='')
                else:
                    print(f'{key}: {None} / ', end='')
                if test_acc_regime[key] is not None:
                    print(f'{test_acc_regime[key]: .3f}')
                else:
                    print(None)
