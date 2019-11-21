#!/user/bin/env python3
# -*- utf-8 -*-
# author shengqiong.wu

from __future__ import division
import logging
import os
import random
import tqdm

import torch
import torchtext
from torch import optim
import torch.nn as nn
from sklearn.metrics import classification_report
from sklearn import metrics

import seq2seq
from seq2seq.evaluator.multi_task_evaluator import Evaluator
from seq2seq.loss.loss import NLLLoss
from seq2seq.optim.optim import Optimizer
from seq2seq.util.checkpoint import Checkpoint


class MultiTaskTrainer(object):
    """ The SupervisedTrainer class helps in setting up a training framework in a
    supervised setting.
    Args:
        expt_dir (optional, str): experiment Directory to store details of the experiment,
            by default it makes a folder in the current directory to store the details (default: `experiment`).
        loss (seq2seq.loss.loss.Loss, optional): loss for training, (default: seq2seq.loss.NLLLoss)
        batch_size (int, optional): batch size for experiment, (default: 64)
        checkpoint_every (int, optional): number of batches to checkpoint after, (default: 100)
    """
    def __init__(self, expt_dir='experiment', loss=NLLLoss(), batch_size=64,
                 random_seed=123,
                 checkpoint_every=100, print_every=100):
        self._trainer = "Simple Trainer"
        self.random_seed = random_seed
        if random_seed is not None:
            random.seed(random_seed)
            torch.manual_seed(random_seed)
        self.loss = loss
        self.class_loss = nn.CrossEntropyLoss()
        self.evaluator = Evaluator(loss=self.loss, batch_size=batch_size)
        self.optimizer = None
        self.checkpoint_every = checkpoint_every
        self.print_every = print_every

        if not os.path.isabs(expt_dir):
            expt_dir = os.path.join(os.getcwd(), expt_dir)
        self.expt_dir = expt_dir
        if not os.path.exists(self.expt_dir):
            os.makedirs(self.expt_dir)
        self.batch_size = batch_size

        self.logger = logging.getLogger(__name__)

    def _train_batch(self, input_variable, input_lengths, target_variable, model, teacher_forcing_ratio,
                     train_type=None):
        if train_type == 'pre_train':
            loss = self.loss
            # Forward propagation
            (decoder_outputs, decoder_hidden, other) = model(input_variable, input_lengths,
                                                             target_variable,
                                                             teacher_forcing_ratio=teacher_forcing_ratio,
                                                             train_type=train_type)
            # Get loss
            loss.reset()
            for step, step_output in enumerate(decoder_outputs):
                batch_size = target_variable.size(0)
                loss.eval_batch(step_output.contiguous().view(batch_size, -1), target_variable[:, step + 1])
            # Backward propagation

            model.zero_grad()
            loss.backward()
            self.optimizer.step()

            return loss.get_loss()
        else:
            class_result = model(input_variable, input_lengths,
                                 target_variable,
                                 teacher_forcing_ratio=teacher_forcing_ratio)
            c_loss = self.class_loss(class_result[0], (target_variable - 2).squeeze(1))
            model.zero_grad()
            c_loss.backward()
            self.optimizer.step()

            return c_loss.cpu().item()

    def _pre_train_epochs(self, data, model, n_epochs, start_epoch, start_step,
                          dev_data=None, test_data=None, teacher_forcing_ratio=0.6):
        log = self.logger

        print_loss_total = 0  # Reset every print_every
        epoch_loss_total = 0  # Reset every epoch

        device = None if torch.cuda.is_available() else -1
        batch_iterator = torchtext.data.BucketIterator(
            dataset=data, batch_size=self.batch_size,
            sort=False, sort_within_batch=True,
            sort_key=lambda x: len(x.norm_src),
            device=device, repeat=False)

        steps_per_epoch = len(batch_iterator)
        total_steps = steps_per_epoch * n_epochs

        step = start_step
        step_elapsed = 0
        min_loss = float("inf")
        early_stop = 0
        for epoch in range(start_epoch, n_epochs + 1):
            log.debug("Epoch: %d, Step: %d" % (epoch, step))

            batch_generator = batch_iterator.__iter__()
            # consuming seen batches from previous training
            for _ in range((epoch - 1) * steps_per_epoch, step):
                next(batch_generator)


            model.train(True)
            for batch in batch_generator:
                step += 1
                step_elapsed += 1

                input_variables, input_lengths = getattr(batch, seq2seq.norm_src_field_name)
                target_variables = getattr(batch, seq2seq.norm_tgt_field_name)

                if torch.cuda.is_available():
                    input_variables = input_variables.cuda()
                    target_variables = target_variables.cuda()

                loss = self._train_batch(input_variables,
                                         input_lengths.tolist(), target_variables,
                                         model, teacher_forcing_ratio, 'pre_train')

                # Record average loss
                print_loss_total += loss
                epoch_loss_total += loss

                if step % self.print_every == 0 and step_elapsed > self.print_every:
                    print_loss_avg = print_loss_total / self.print_every
                    print_loss_total = 0
                    log_msg = 'Progress: %d%%, Train %s: %.4f' % (
                        step / total_steps * 100,
                        self.loss.name,
                        print_loss_avg)
                    log.info(log_msg)

            if step_elapsed == 0:
                continue

            epoch_loss_avg = epoch_loss_total / min(steps_per_epoch, step - start_step)
            epoch_loss_total = 0
            log_msg = "Finished epoch %d: Train %s: %.4f" % (epoch, self.loss.name, epoch_loss_avg)
            if dev_data is not None:
                dev_loss, accuracy = self.evaluator.pre_train_evaluator(model, dev_data)
                self.optimizer.update(dev_loss, epoch)

                if dev_loss < min_loss:
                    Checkpoint(model=model,
                               optimizer=self.optimizer,
                               epoch=epoch, step=step,
                               input_vocab=data.fields[seq2seq.norm_src_field_name].vocab,
                               output_vocab=data.fields[seq2seq.norm_tgt_field_name].vocab).save(self.expt_dir)
                log_msg += ", Dev %s: %.4f, Normlization Accuracy: %.4f" % (self.loss.name, dev_loss, accuracy)

                model.train(mode=True)
            else:
                self.optimizer.update(epoch_loss_avg, epoch)

            log.info(log_msg)

    def _train_epoches(self, data, model, n_epochs, start_epoch, start_step,
                       dev_data=None, test_data=None, teacher_forcing_ratio=0):
        log = self.logger

        print_loss_total = 0  # Reset every print_every
        epoch_loss_total = 0  # Reset every epoch

        device = None if torch.cuda.is_available() else -1
        batch_iterator = torchtext.data.BucketIterator(
            dataset=data, batch_size=self.batch_size,
            sort=False, sort_within_batch=True,
            sort_key=lambda x: len(x.class_src),
            device=device, repeat=False,
        )

        steps_per_epoch = len(batch_iterator)
        total_steps = steps_per_epoch * n_epochs

        step = start_step
        step_elapsed = 0
        min_f1 = 0
        for epoch in range(start_epoch, n_epochs + 1):
            log.debug("Epoch: %d, Step: %d" % (epoch, step))

            batch_generator = batch_iterator.__iter__()
            # consuming seen batches from previous training
            for _ in range((epoch - 1) * steps_per_epoch, step):
                next(batch_generator)

            model.train(True)
            for batch in batch_generator:
                step += 1
                step_elapsed += 1

                input_variables, class_lengths = getattr(batch, seq2seq.class_src_field_name)
                target_variables, _ = getattr(batch, seq2seq.class_tgt_field_name)

                if torch.cuda.is_available():
                    input_variables = input_variables.cuda()
                    target_variables = target_variables.cuda()

                loss = self._train_batch(input_variables,
                                         class_lengths, target_variables,
                                         model, teacher_forcing_ratio)

                # Record average loss
                print_loss_total += loss
                epoch_loss_total += loss

                if step % self.print_every == 0 and step_elapsed > self.print_every:
                    print_loss_avg = print_loss_total / self.print_every
                    print_loss_total = 0
                    log_msg = 'Progress: %d%%, Train average loss/%d : %.4f' % (
                        step / total_steps * 100,
                        self.print_every,
                        print_loss_avg)
                    log.info(log_msg)

            if step_elapsed == 0:
                continue

            epoch_loss_avg = epoch_loss_total / min(steps_per_epoch, step - start_step)
            epoch_loss_total = 0
            log_msg = "Finished epoch %d: Train epoch average loss: %.4f" % (epoch, epoch_loss_avg)
            if dev_data is not None:
                pred_result, gold_result = self.evaluator.evaluate(model, dev_data)
                # self.optimizer.update(dev_loss, epoch)

                f1 = metrics.f1_score(gold_result, pred_result, average='weighted')
                p = metrics.precision_score(gold_result, pred_result, average='weighted')
                r = metrics.recall_score(gold_result, pred_result, average='weighted')
                acc = metrics.accuracy_score(gold_result, pred_result)

                if f1 > min_f1:
                    Checkpoint(model=model,
                               optimizer=self.optimizer,
                               epoch=epoch, step=step,
                               input_vocab=data.fields[seq2seq.class_src_field_name].vocab,
                               output_vocab=data.fields[seq2seq.class_tgt_field_name].vocab).save(self.expt_dir)
                # log_msg += ", Dev %s: %.4f, Normlization Accuracy: %.4f" % (self.loss.name, dev_loss, accuracy)
                log_msg += '\nAggressive language detection ,weighted_f1: {}, p: {}, r: {}, acc: {}'.format(f1, p, r, acc)
                model.train(mode=True)
            else:
                pass
                # self.optimizer.update(epoch_loss_avg, epoch)

            log.info(log_msg)

    def train(self, model, data, num_epochs=5,
              resume=False, dev_data=None,
              optimizer=None, teacher_forcing_ratio=0, lr=0.003, pre_train=False):
        """ Run training for a given model.
        Args:
            model (seq2seq.models): model to run training on, if `resume=True`, it would be
               overwritten by the model loaded from the latest checkpoint.
            data (seq2seq.dataset.dataset.Dataset): dataset object to train on
            num_epochs (int, optional): number of epochs to run (default 5)
            resume(bool, optional): resume training with the latest checkpoint, (default False)
            dev_data (seq2seq.dataset.dataset.Dataset, optional): dev Dataset (default None)
            optimizer (seq2seq.optim.Optimizer, optional): optimizer for training
               (default: Optimizer(pytorch.optim.Adam, max_grad_norm=5))
            teacher_forcing_ratio (float, optional): teaching forcing ratio (default 0)
        Returns:
            model (seq2seq.models): trained model.
        """
        # If training is set to resume
        if resume:
            latest_checkpoint_path = Checkpoint.get_latest_checkpoint(self.expt_dir)
            resume_checkpoint = Checkpoint.load(latest_checkpoint_path)
            model = resume_checkpoint.model
            self.optimizer = resume_checkpoint.optimizer

            # A walk around to set optimizing parameters properly
            resume_optim = self.optimizer.optimizer
            defaults = resume_optim.param_groups[0]
            defaults.pop('params', None)
            defaults.pop('initial_lr', None)
            self.optimizer.optimizer = resume_optim.__class__(model.parameters(), **defaults)

            start_epoch = resume_checkpoint.epoch
            step = resume_checkpoint.step
        else:
            start_epoch = 1
            step = 0
            if optimizer is None:
                optimizer = Optimizer(optim.Adam(model.parameters(), lr=lr), max_grad_norm=5)
            self.optimizer = optimizer

        self.logger.info("Optimizer: %s, Scheduler: %s" % (self.optimizer.optimizer, self.optimizer.scheduler))

        if pre_train:
            self._pre_train_epochs(data, model, num_epochs, start_epoch, step, dev_data=dev_data,
                                   teacher_forcing_ratio=teacher_forcing_ratio)
        else:
            self._train_epoches(data, model, num_epochs,
                                start_epoch, step, dev_data=dev_data,
                                teacher_forcing_ratio=teacher_forcing_ratio)

        return model
