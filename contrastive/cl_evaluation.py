import logging
import math
import pickle as pkl
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.contrastive_dataset import ContrastiveResponseSelectionDataset
from models import Model
from models.utils.checkpointing import load_checkpoint
from models.utils.scorer import calculate_candidates_ranking, logits_mrr, \
    logits_recall_at_k, precision_at_one, mean_average_precision


class ContrastiveEvaluation(object):
    def __init__(self, hparams, model=None):

        self.hparams = hparams
        self.model = model
        self._logger = logging.getLogger(__name__)
        self.device = (torch.device("cuda", self.hparams.gpu_ids[0])
                       if self.hparams.gpu_ids[0] >= 0 else torch.device("cpu"))
        self.split = hparams.evaluate_data_type
        print("Evaluation Split :", self.split)
        do_valid, do_test, do_train = False, False, False

        if self.split == "dev":
            do_valid = True
        elif self.split == 'test':
            do_test = True
        elif self.split == 'train':
            do_train = True
        self._build_dataloader(do_valid=do_valid, do_test=do_test, do_train=do_train)
        if self.split == 'dev':
            self._dataloader = self.valid_dataloader
        elif self.split == 'test':
            self._dataloader = self.test_dataloader
        elif self.split == 'train':
            self._dataloader = self.train_dataloader
        else:
            raise ValueError('Split error while loading data')

        if model is None:
            print("No pre-defined model!")
            self._build_model()

    def _build_dataloader(self, do_valid=False, do_test=False, do_train=False):

        if do_train:
            self.train_dataset = ContrastiveResponseSelectionDataset(
                self.hparams,
                split="train",
            )
            self.train_dataloader = DataLoader(
                self.train_dataset,
                batch_size=self.hparams.eval_batch_size,
                num_workers=self.hparams.cpu_workers,
                drop_last=False,
                shuffle=False,
                collate_fn=ContrastiveResponseSelectionDataset.collate_fn,
            )

        if do_valid:
            self.valid_dataset = ContrastiveResponseSelectionDataset(
                self.hparams,
                split="dev",
            )
            self.valid_dataloader = DataLoader(
                self.valid_dataset,
                batch_size=self.hparams.eval_batch_size,
                num_workers=self.hparams.cpu_workers,
                drop_last=False,
                collate_fn=ContrastiveResponseSelectionDataset.collate_fn,
            )

        if do_test:
            self.test_dataset = ContrastiveResponseSelectionDataset(
                self.hparams,
                split="test",
            )

            self.test_dataloader = DataLoader(
                self.test_dataset,
                batch_size=self.hparams.eval_batch_size,
                num_workers=self.hparams.cpu_workers,
                drop_last=False,
                collate_fn=ContrastiveResponseSelectionDataset.collate_fn,
            )

    def _build_model(self):
        self.model = Model(self.hparams)
        self.model = self.model.to(self.device)
        # Use Multi-GPUs
        if -1 not in self.hparams.gpu_ids and len(self.hparams.gpu_ids) > 1:
            self.model = nn.DataParallel(self.model, self.hparams.gpu_ids)

    def run_evaluate(self, evaluation_path):
        self._logger.info("Evaluation")
        model_state_dict, optimizer_state_dict = load_checkpoint(evaluation_path)
        print(evaluation_path)
        if isinstance(self.model, nn.DataParallel):
            self.model.module.load_state_dict(model_state_dict)
        else:
            self.model.load_state_dict(model_state_dict)

        k_list = self.hparams.recall_k_list
        total_mrr, total_prec_at_one, total_map = 0, 0, 0
        total_examples, total_correct = 0, 0

        self.model.eval()

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self._dataloader)):
                buffer_batch = batch.copy()
                for key in buffer_batch["res_sel"]:
                    buffer_batch["res_sel"][key] = buffer_batch["res_sel"][key].to(self.device)
                buffer_batch_dict = {'original': buffer_batch}

                logits, loss = self.model(buffer_batch_dict)
                pred = torch.sigmoid(logits).to("cpu").tolist()

                rank_by_pred, pos_index, stack_scores = \
                    calculate_candidates_ranking(np.array(pred),
                                                 np.array(buffer_batch["res_sel"]["label"].to("cpu").tolist()),
                                                 self.hparams.evaluate_candidates_num)

                num_correct = logits_recall_at_k(pos_index, k_list)

                if self.hparams.task_name in ["douban", "kakao"]:
                    total_prec_at_one += precision_at_one(rank_by_pred)
                    total_map += mean_average_precision(pos_index)
                    for pred in rank_by_pred:
                        if sum(pred) == 0:
                            total_examples -= 1

                total_mrr += logits_mrr(pos_index)

                total_correct = np.add(total_correct, num_correct)
                total_examples += math.ceil(
                    buffer_batch["res_sel"]["label"].size()[0] / self.hparams.evaluate_candidates_num)

                recall_result = ""
                if (batch_idx + 1) % self.hparams.evaluate_print_step == 0:
                    for i in range(len(k_list)):
                        recall_result += "Recall@%s : " % k_list[i] + "%.2f%% | " % (
                                (total_correct[i] / total_examples) * 100)
                    else:
                        print("%d[th] | %s | MRR : %.3f | P@1 : %.3f | MAP : %.3f" %
                              (batch_idx + 1, recall_result, float(total_mrr / total_examples),
                               float(total_prec_at_one / total_examples), float(total_map / total_examples)))
                    self._logger.info("%d[th] | %s | MRR : %.3f | P@1 : %.3f | MAP : %.3f" %
                                      (batch_idx + 1, recall_result, float(total_mrr / total_examples),
                                       float(total_prec_at_one / total_examples), float(total_map / total_examples)))

            avg_mrr = float(total_mrr / total_examples)
            avg_prec_at_one = float(total_prec_at_one / total_examples)
            avg_map = float(total_map / total_examples)
            recall_result = ""

            for i in range(len(k_list)):
                recall_result += "Recall@%s : " % k_list[i] + "%.2f%% | " % ((total_correct[i] / total_examples) * 100)
            print(recall_result)
            print("MRR: %.4f" % avg_mrr)
            print("P@1: %.4f" % avg_prec_at_one)
            print("MAP: %.4f" % avg_map)

            self._logger.info(recall_result)
            self._logger.info("MRR: %.4f" % avg_mrr)
            self._logger.info("P@1: %.4f" % avg_prec_at_one)
            self._logger.info("MAP: %.4f" % avg_map)

        return [total_correct[i] / total_examples for i in range(len(k_list))]

    def dump_logits(self, best_model_path):
        model_state_dict, optimizer_state_dict = load_checkpoint(best_model_path)
        print(f'Load model from {best_model_path}')
        if isinstance(self.model, nn.DataParallel):
            self.model.module.load_state_dict(model_state_dict)
        else:
            self.model.load_state_dict(model_state_dict)

        self.model.eval()
        self.model.return_augment = True
        ret_logits = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self._dataloader)):
                buffer_batch = batch.copy()
                for group in buffer_batch:
                    group_data = buffer_batch[group]
                    for task_key in group_data:
                        for key in group_data[task_key]:
                            buffer_batch[group][task_key][key] = buffer_batch[group][task_key][key].to(self.device)

                (logits, aug_logits), loss = self.model(buffer_batch)
                pred = torch.sigmoid(logits).to("cpu").tolist()
                aug_pred = torch.sigmoid(aug_logits).to("cpu").tolist()
                ret_logits += list(zip(pred, aug_pred))

        pkl.dump(ret_logits, open(f'cache/{self.hparams.task_name}_soft_logits_{datetime.now().strftime("%m%d%H%M%S")}.pkl', 'wb'))


    def run_evaluate_with_data(self, evaluation_path, data):
        self._logger.info("Evaluation")
        ret = []

        test_dataset = ContrastiveResponseSelectionDataset(
            self.hparams,
            split="",
            data=data,
        )

        test_dataloader = DataLoader(
            test_dataset,
            batch_size=128,
            num_workers=self.hparams.cpu_workers,
            drop_last=False,
        )

        model_state_dict, optimizer_state_dict = load_checkpoint(evaluation_path)
        print(evaluation_path)
        if isinstance(self.model, nn.DataParallel):
            self.model.module.load_state_dict(model_state_dict)
        else:
            self.model.load_state_dict(model_state_dict)

        self.model.eval()

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(test_dataloader)):
                buffer_batch = batch[0].copy()
                for key in buffer_batch["res_sel"]:
                    buffer_batch["res_sel"][key] = buffer_batch["res_sel"][key].to(self.device)
                buffer_batch_dict = {'original': buffer_batch}
                logits, loss = self.model(buffer_batch_dict)
                pred = torch.sigmoid(logits).to("cpu").tolist()
                ret += pred

        return ret
