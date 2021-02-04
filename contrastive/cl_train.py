import logging
import os
import random
from datetime import datetime

import numpy as np
import torch
from torch import nn, optim
from torch.cuda import amp
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup

from contrastive.cl_evaluation import ContrastiveEvaluation
from data.contrastive_dataset_v6 import ContrastiveResponseSelectionDataset
from models import Model
from models.utils.checkpointing import CheckpointManager, load_checkpoint
import global_variables


class ContrastiveResponseSelection(object):
    def __init__(self, hparams):
        self.hparams = hparams
        self._logger = logging.getLogger(__name__)

        random.seed(hparams.random_seed)
        np.random.seed(hparams.random_seed)
        torch.manual_seed(hparams.random_seed)
        torch.cuda.manual_seed_all(hparams.random_seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def _build_dataloader(self):
        # =============================================================================
        #   SETUP DATASET, DATALOADER
        # =============================================================================
        self.train_dataset = ContrastiveResponseSelectionDataset(self.hparams, split="train")
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.hparams.train_batch_size,
            num_workers=self.hparams.cpu_workers,
            shuffle=True,
            drop_last=True,
            collate_fn=ContrastiveResponseSelectionDataset.collate_fn
        )

        print("""
       # -------------------------------------------------------------------------
       #   DATALOADER FINISHED
       # -------------------------------------------------------------------------
       """)

    def _build_model(self):
        # =============================================================================
        #   MODEL : Standard, Mention Pooling, Entity Marker
        # =============================================================================
        print('\t* Building model...')

        self.model = Model(self.hparams)
        self.model = self.model.to(self.device)

        # Use Multi-GPUs
        if -1 not in self.hparams.gpu_ids and len(self.hparams.gpu_ids) > 1:
            self.model = nn.DataParallel(self.model, self.hparams.gpu_ids)

        # =============================================================================
        #   CRITERION
        # =============================================================================

        self.iterations = len(self.train_dataset) // self.hparams.virtual_batch_size

        # Prepare optimizer and schedule (linear warmup and decay)
        if self.hparams.optimizer_type == "Adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.hparams.learning_rate)
        elif self.hparams.optimizer_type == "AdamW":
            no_decay = ['bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                 'weight_decay': self.hparams.weight_decay},
                {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0}
            ]
            self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate,
                                   eps=self.hparams.adam_epsilon)
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer, num_warmup_steps=self.hparams.warmup_steps,
                num_training_steps=self.iterations * self.hparams.num_epochs)
        self.scaler = amp.GradScaler()
        self.cl_loss_ratio = self.hparams.cl_loss_ratio

    def _setup_training(self):
        if self.hparams.save_dirpath == 'checkpoints/':
            self.save_dirpath = os.path.join(self.hparams.root_dir, self.hparams.save_dirpath)
        self.summary_writer = SummaryWriter(self.save_dirpath)
        self.checkpoint_manager = CheckpointManager(self.model, self.optimizer, self.save_dirpath, hparams=self.hparams)

        # If loading from checkpoint, adjust start epoch and load parameters.
        if self.hparams.load_pthpath == "":
            self.start_epoch = 1
        else:
            # "path/to/checkpoint_xx.pth" -> xx
            self.start_epoch = int(self.hparams.load_pthpath.split("_")[-1][:-4])
            self.start_epoch += 1
            load_pthpath = os.path.join(self.hparams.bert_pretrained_dir, self.hparams.bert_pretrained, self.hparams.load_pthpath)
            model_state_dict, optimizer_state_dict = load_checkpoint(load_pthpath)
            model_state_dict = {k:v for k, v in model_state_dict.items() if k.startswith('_model')}
            if isinstance(self.model, nn.DataParallel):
                current_state_dict = self.model.module.state_dict()
                current_state_dict.update(model_state_dict)
                self.model.module.load_state_dict(current_state_dict)
            else:
                current_state_dict = self.model.state_dict()
                current_state_dict.update(model_state_dict)
                self.model.load_state_dict(current_state_dict)
            # self.optimizer.load_state_dict(optimizer_state_dict)
            self.previous_model_path = self.hparams.load_pthpath
            print("Loaded model from {}".format(self.hparams.load_pthpath))

        print(
            """
            # -------------------------------------------------------------------------
            #   Setup Training Finished
            # -------------------------------------------------------------------------
            """
        )

    def train(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._build_dataloader()
        self._build_model()
        self._setup_training()

        global_variables.num_iter = len(self.train_dataset) // self.hparams.virtual_batch_size
        global_variables.epoch = 0

        # ins, del, mod check!

        # Evaluation Setup
        evaluation = ContrastiveEvaluation(self.hparams, model=self.model)

        start_time = datetime.now().strftime('%H:%M:%S')
        self._logger.info("Start train model at %s" % start_time)

        train_begin = datetime.utcnow()  # New
        global_iteration_step = 0
        accu_loss, accu_res_sel_loss, accu_cl_loss, accu_ins_loss = 0, 0, 0, 0
        accu_cnt = 0

        best_recall_list, best_model_path = [0], ''

        for epoch in range(self.start_epoch, self.hparams.num_epochs + 1):
            global_variables.epoch = epoch
            self.model.train()
            tqdm_batch_iterator = tqdm(self.train_dataloader)
            accu_batch = 0
            for batch_idx, batch in enumerate(tqdm_batch_iterator):

                buffer_batch = batch.copy()
                for group in buffer_batch:
                    for task_key in buffer_batch[group]:
                        for key in buffer_batch[group][task_key]:
                            buffer_batch[group][task_key][key] = buffer_batch[group][task_key][key].to(self.device)

                buffer_batch = batch
                with amp.autocast():
                    _, losses = self.model(buffer_batch)
                res_sel_loss, contrastive_loss, ins_loss = losses
                if res_sel_loss is not None:
                    res_sel_loss = self.hparams.res_sel_loss_ratio * res_sel_loss.mean()
                    accu_res_sel_loss += res_sel_loss.item()
                loss = self.scaler.scale(res_sel_loss)
                if self.hparams.do_contrastive:
                    cl_loss = self.cl_loss_ratio * contrastive_loss.mean()
                    accu_cl_loss += cl_loss.item()
                    cl_loss = self.scaler.scale(cl_loss)
                    loss += cl_loss
                if self.hparams.do_sent_insertion:
                    ins_loss = ins_loss.mean()
                    accu_ins_loss += ins_loss.item()
                    ins_loss = self.scaler.scale(ins_loss)
                    loss += ins_loss

                loss.backward()
                accu_loss += loss.item()
                accu_cnt += 1

                # TODO: virtual batch implementation
                accu_batch += buffer_batch["original"]["res_sel"]["label"].shape[0]

                if self.hparams.virtual_batch_size == accu_batch \
                        or batch_idx == (len(self.train_dataset) // self.hparams.train_batch_size):  # last batch

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    if self.hparams.optimizer_type == "AdamW":
                        self.scheduler.step()

                    # nn.utils.clip_grad_norm_(self.model.parameters(), self.hparams.max_gradient_norm)
                    self.optimizer.zero_grad()

                    accu_batch = 0

                    global_iteration_step += 1
                    global_variables.global_step += 1
                    # description = "[{}][Epoch: {:3d}][Iter: {:6d}][Loss: {:6f}][Res_Loss: {:4f}]" \
                    #               "[Ins_Loss: {:4f}][Del_Loss: {:4f}][Srch_Loss: {:4f}][lr: {:7f}]".format(
                    #     datetime.utcnow() - train_begin,
                    #     epoch,
                    #     global_iteration_step, accu_loss / accu_cnt,
                    #     accu_res_sel_loss / accu_cnt, accu_ins_loss / accu_cnt, accu_del_loss / accu_cnt,
                    #     accu_srch_loss / accu_cnt,
                    #     self.optimizer.param_groups[0]['lr'])
                    description = "[Epoch:{:2d}][Iter:{:3d}][Loss: {:.2e}][Res/CL/Ins: {:.2e}/{:.2e}/{:.2e}][lr: {:.2e}]".format(
                        epoch,
                        global_iteration_step, accu_loss / accu_cnt, accu_res_sel_loss / accu_cnt,
                        accu_cl_loss / accu_cnt, accu_ins_loss / accu_cnt,
                        self.optimizer.param_groups[0]['lr'])
                    tqdm_batch_iterator.set_description(description)

                    # tensorboard
                    if global_iteration_step % self.hparams.tensorboard_step == 0:
                        self._logger.info(description)
                        accu_loss, accu_cl_loss, accu_res_sel_loss, accu_ins_loss, accu_cnt = 0, 0, 0, 0, 0

                # if batch_idx == len(self.train_dataloader) // 2:
                #     recall_list = evaluation.run_evaluate(self.model)
                #     if recall_list[0] > best_recall_list[0]:
                #         best_recall_list = recall_list
                #         state_dict = self.model.module.state_dict() if isinstance(self.model, nn.DataParallel) else self.model.state_dict()
                #         torch.save(state_dict, os.path.join(self.checkpoint_manager.ckpt_dirpath, 'best.pt'))
                #     self.model.train()

            self.checkpoint_manager.step(epoch)
            self.previous_model_path = os.path.join(self.checkpoint_manager.ckpt_dirpath, "checkpoint_%d.pth" % (epoch))
            self._logger.info(self.previous_model_path)

            torch.cuda.empty_cache()
            self._logger.info("Evaluation after %d epoch" % epoch)
            recall_list = evaluation.run_evaluate(self.previous_model_path)
            if recall_list[0] > best_recall_list[0]:
                best_recall_list = recall_list
                best_model_path = self.previous_model_path
                state_dict = self.model.module.state_dict() if isinstance(self.model, nn.DataParallel) else self.model.state_dict()
                torch.save(state_dict, os.path.join(self.checkpoint_manager.ckpt_dirpath, 'best.pt'))
            torch.cuda.empty_cache()

        print(f'Best recalls: {best_recall_list}, model path: {best_model_path}')
        return best_recall_list, best_model_path
