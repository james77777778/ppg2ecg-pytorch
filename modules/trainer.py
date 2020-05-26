import os
import sys
import json
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from absl import flags

from modules.data import datasets
from modules.models import (PPG2ECG_BASELINE_LSTM, PPG2ECG)
from modules.loss import QRSLoss
from modules.util import signal2waveform, dict2table


FLAGS = flags.FLAGS


class Trainer(object):
    def __init__(self):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.run_name = FLAGS.run_name
        self.log_path = os.path.join(FLAGS.logdir, self.run_name)
        self.save_path = os.path.join(FLAGS.logdir, self.run_name, 'model')
        os.makedirs(self.save_path, exist_ok=True)
        if FLAGS.seed != "":
            torch.manual_seed(int(FLAGS.seed))
            np.random.seed(int(FLAGS.seed))

        # tbwriter
        self.tbwriter = SummaryWriter(log_dir=self.log_path)
        flags_dict = dict(
            (item.name, item.value) for item
            in FLAGS.get_key_flags_for_module(sys.argv[0]))
        print(json.dumps(flags_dict, indent=4))
        self.tbwriter.add_text(self.run_name, dict2table(flags_dict), 0)

        # training parameters
        self.epoch = FLAGS.epoch
        self.eval_step = FLAGS.eval_step
        self.save_step = FLAGS.save_step

        # dataset
        dataset_class = datasets[FLAGS.dataset]
        train_set = dataset_class(sigma=FLAGS.qrs_sigma, use_aug=FLAGS.aug)
        test_set = dataset_class(sigma=FLAGS.qrs_sigma, is_train=False,
                                 use_aug=False)
        visual_set = random.choices(test_set, k=30)
        self.train_loader = DataLoader(
            dataset=train_set,
            batch_size=FLAGS.batch_size,
            shuffle=True,
            num_workers=3,
            pin_memory=True)
        self.test_loader = DataLoader(dataset=test_set, batch_size=1)
        self.visual_loader = DataLoader(dataset=visual_set, batch_size=1)

        # model
        if FLAGS.lstm:
            self.model = PPG2ECG_BASELINE_LSTM(50).to(self.device)
        else:
            self.model = PPG2ECG(FLAGS.input_size,
                                 use_stn=FLAGS.stn,
                                 use_attention=FLAGS.attn).to(self.device)
        # optim
        self.opt = optim.Adam(self.model.parameters(), lr=FLAGS.lr)
        # qrsloss
        if FLAGS.qrsloss:
            self.qrsloss = QRSLoss(beta=FLAGS.qrs_beta)
        else:
            self.qrsloss = None
        # learning rate scheduler
        if FLAGS.lr_sched:
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.opt, FLAGS.epoch//10)
        else:
            self.scheduler = None

    def train(self, batch, iter):
        ppg, ecg = batch['ppg'].to(self.device), batch['ecg'].to(self.device)
        exp_rpeaks = batch['exp_rpeaks'].to(self.device)
        loss = {}
        self.opt.zero_grad()
        out_dict = self.model(ppg)
        output = out_dict['output']
        l1_loss = F.l1_loss(output, ecg)
        loss['train/L1'] = l1_loss.item()
        if self.qrsloss is not None:
            qrs_loss = self.qrsloss(output, ecg, exp_rpeaks)
            loss['train/QRS_L1'] = qrs_loss.item()
            qrs_loss.backward()
        else:
            l1_loss.backward()
        self.opt.step()
        # write scalar
        for name, value in loss.items():
            self.tbwriter.add_scalar(name, value, iter)

    def test(self, best, epoch, iter):
        self.model.eval()
        running_loss = 0
        running_qrs_loss = 0
        len_test_set = len(self.test_loader.dataset)
        pbar = tqdm(self.test_loader,
                    dynamic_ncols=True,
                    desc='Testing')
        for batch in pbar:
            ppg = batch['ppg'].to(self.device)
            ecg = batch['ecg'].to(self.device)
            exp_rpeaks = batch['exp_rpeaks'].to(self.device)
            loss = {}
            with torch.no_grad():
                out_dict = self.model(ppg)
                output = out_dict['output']
                l1_loss = F.l1_loss(output, ecg)
                loss['test/L1'] = l1_loss.item()
                if self.qrsloss is not None:
                    qrs_loss = self.qrsloss(output, ecg, exp_rpeaks)
                    loss['test/QRS_L1'] = qrs_loss.item()
            running_loss += loss['test/L1']
            running_qrs_loss += loss['test/QRS_L1']
        pbar.close()
        # write scalar
        running_loss /= len_test_set
        running_qrs_loss /= len_test_set
        self.tbwriter.add_scalar('test/L1', running_loss, iter)
        self.tbwriter.add_scalar('test/QRS_L1', running_qrs_loss, iter)
        # save best_checkpoint
        if running_loss < best["L1"]:
            best["L1"] = running_loss
            self.tbwriter.add_scalar('test/best_L1', running_loss, iter)
            if self.qrsloss is None:
                self.best_checkpoint = {
                    'epoch': epoch+1,
                    'net': self.model.state_dict(),
                }
        if running_qrs_loss < best["QRS_L1"]:
            best["QRS_L1"] = running_qrs_loss
            self.tbwriter.add_scalar(
                'test/best_QRS_L1', running_qrs_loss, iter)
            if self.qrsloss is not None:
                self.best_checkpoint = {
                    'epoch': epoch+1,
                    'net': self.model.state_dict(),
                }
        self.model.train()

    def visualize(self, epoch, iter):
        self.model.eval()
        pbar = tqdm(self.visual_loader,
                    dynamic_ncols=True,
                    desc='Visualizing')
        for batch in pbar:
            with torch.no_grad():
                ppg = batch['ppg'].to(self.device)
                ecg = batch['ecg'].to(self.device)
                path = os.path.basename(batch['path'][0])
                out_dict = self.model(ppg)
                output = out_dict['output']
                output_stn = out_dict['output_stn']
                ppg_waveform = signal2waveform(ppg.squeeze().cpu().numpy())
                ecg_waveform = signal2waveform(ecg.squeeze().cpu().numpy())
                out_waveform = signal2waveform(output.squeeze().cpu().numpy())
                out_stn_waveform = signal2waveform(
                    output_stn.squeeze().cpu().numpy())
                concat_img = np.concatenate([
                    ppg_waveform, np.ones((ppg_waveform.shape[0], 25)),
                    out_stn_waveform, np.ones((ppg_waveform.shape[0], 25)),
                    out_waveform, np.ones((ppg_waveform.shape[0], 25)),
                    ecg_waveform
                ], axis=1)
                self.tbwriter.add_image(
                    'waveform/{}'.format(path), concat_img, iter,
                    dataformats='HW'
                )
        pbar.close()

    def save_checkpoint(self, checkpoint, save_path, save_name):
        save_path = os.path.join(save_path, 'model_{}.pth'.format(save_name))
        torch.save(checkpoint, save_path)

    def run(self):
        best = {'L1': 100000, 'QRS_L1': 100000}
        self.best_checkpoint = None
        iters = 1
        pbar = tqdm(total=self.epoch*len(self.train_loader),
                    dynamic_ncols=True)
        for epoch in range(self.epoch):
            self.tbwriter.add_scalar('train/epoch', epoch+1, iters)
            self.model.train()
            for batch in self.train_loader:
                self.train(batch, iters)
                iters += 1
                pbar.update(1)
            if epoch % self.eval_step == 0 or epoch == self.epoch-1:
                self.test(best, epoch, iters)
                self.visualize(epoch, iters)
            if epoch % self.save_step == 0 or epoch == self.epoch-1:
                self.save_checkpoint(
                    {
                        'epoch': epoch+1,
                        'net': self.model.state_dict()
                    },
                    self.save_path,
                    '{}'.format(epoch+1)
                )
            if self.scheduler:
                self.scheduler.step()
        self.tbwriter.close()
        self.save_checkpoint(self.best_checkpoint, self.save_path, 'best')
