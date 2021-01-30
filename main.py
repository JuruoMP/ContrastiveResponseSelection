import os
import argparse
import collections
import logging
import json
from datetime import datetime

import numpy as np
import torch

from config.hparams import *
from train import ResponseSelection
from post_train.post_training import PostTraining
from contrastive.cl_train import ContrastiveResponseSelection

from evaluation import Evaluation
from contrastive.cl_evaluation import ContrastiveEvaluation
from data.ubuntu_corpus_v1.ubuntu_data_utils import InputExamples

PARAMS_MAP = {
    # Pre-trained Models
    "bert_base": BASE_PARAMS,
    "bert_post": POST_PARAMS,

    "electra_base": BASE_PARAMS,
    "electra_post": ELECTRA_POST_PARAMS,

    "bert_base_eot": BASE_EOT_PARAMS,

    "bert_post_training": BERT_POST_TRAINING_PARAMS,
    "electra_post_training": ELECTRA_POST_TRAINING_PARAMS,
    "electra-nsp_post_training": ELECTRA_NSP_POST_TRAINING_PARAMS
}

DATASET_MAP = {
    "ubuntu": UBUNTU_PARAMS,
    "e-commerce": ECOMMERCE_PARAMS,
    "douban": DOUBAN_PARAMS
}

PRETRAINED_MODEL_MAP = {
    "bert": BERT_MODEL_PARAMS,
    "electra": ELECTRA_MODEL_PARAMS
}

TRAINING_TYPE_MAP = {
    "fine_tuning": ResponseSelection,
    "post_training": PostTraining,
    "contrastive": ContrastiveResponseSelection
}

EVAL_TYPE_MAP = {
    "fine_tuning": Evaluation,
    "contrastive": ContrastiveEvaluation
}

MULTI_TASK_TYPE_MAP = {
    "contras": CONTRASTIVE_PARAMS,
    "aug": AUGMENT_PARAMS,
    "extra": EXTRA_CONTRASTIVE_PARAMS,
}


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if seed == 0:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def init_logger(path: str, hparams):
    if not os.path.exists(path):
        os.makedirs(path)
    hparams = dict(hparams)
    del hparams['pretrained_config']
    del hparams['pretrained_model']
    json.dump(hparams, open(os.path.join(path, 'config.txt'), 'w', encoding='utf-8'), indent=2)

    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.DEBUG)
    debug_fh = logging.FileHandler(os.path.join(path, "debug.log"))
    debug_fh.setLevel(logging.DEBUG)

    info_fh = logging.FileHandler(os.path.join(path, "info.log"))
    info_fh.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    info_formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s')
    debug_formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s | %(lineno)d:%(funcName)s')

    ch.setFormatter(info_formatter)
    info_fh.setFormatter(info_formatter)
    debug_fh.setFormatter(debug_formatter)

    logger.addHandler(ch)
    logger.addHandler(debug_fh)
    logger.addHandler(info_fh)

    return logger


def train_model(args, hparams):
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    root_dir = os.path.join(hparams["root_dir"], args.model, args.task_name, "%s/" % timestamp)
    logger = init_logger(root_dir, hparams)
    logger.info("Hyper-parameters: %s" % str(hparams))
    hparams["root_dir"] = root_dir

    hparams = collections.namedtuple("HParams", sorted(hparams.keys()))(**hparams)
    model = TRAINING_TYPE_MAP[args.training_type](hparams)
    return model.train()


def evaluate_model(args, hparams):
    hparams = collections.namedtuple("HParams", sorted(hparams.keys()))(**hparams)

    model = EVAL_TYPE_MAP[args.training_type](hparams)
    model.run_evaluate(args.evaluate)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description="Utterance Manipulation Strategy : Response Selection (PyTorch)")
    arg_parser.add_argument("--model", dest="model", type=str, default="bert_base", help="Model Name")
    arg_parser.add_argument("--task_name", dest="task_name", type=str, default="ubuntu", help="Task Name")
    arg_parser.add_argument("--task_type", dest="task_type", type=str,
                            default="response_selection",
                            help="response selection | sentence insertion")
    # bert-base-uncased, bert-post-uncased
    arg_parser.add_argument("--root_dir", dest="root_dir", type=str,
                            default="./",
                            help="model train logs, checkpoints")
    arg_parser.add_argument("--data_dir", dest="data_dir", type=str, required=True,
                            help="training pkl path | h5py files")  # ubuntu_train.pkl, ubuntu_valid_pkl, ubuntu_test.pkl
    arg_parser.add_argument("--bert_pretrained_dir", dest="bert_pretrained_dir", type=str,
                            default="./resources",
                            help="bert pretrained directory")
    arg_parser.add_argument("--bert_pretrained", dest="bert_pretrained", type=str,
                            default="bert-base-uncased",
                            help="bert pretrained directory")  # bert-base-uncased, bert-post-uncased
    arg_parser.add_argument("--bert_checkpoint_path", dest="bert_checkpoint_path", type=str,
                            default="bert-base-uncased-pytorch_model.bin",
                            help="bert pretrained directory")  # bert-base-uncased, bert-post-uncased
    arg_parser.add_argument("--evaluate", dest="evaluate", type=str,
                            help="Evaluation Checkpoint", default="")
    arg_parser.add_argument("--training_type", dest="training_type", type=str, default="fine_tuning",
                            help="fine_tuning or post_training")
    arg_parser.add_argument("--multi_task_type", dest="multi_task_type", type=str, default="",
                            help="contras,hinge,extra")
    arg_parser.add_argument("--gpu_ids", dest="gpu_ids", type=str,
                            help="gpu_ids", default="0")
    arg_parser.add_argument("--electra_gen_config", dest="electra_gen_config", type=str,
                            help="electra_gen_config", default="")  # electra-base-gen, electra-base-chinese-gen
    arg_parser.add_argument("--use_batch_negative", dest="use_batch_negative", type=bool, default=False,
                            help="Use all examples in the batch as negative for contrastive learning")
    arg_parser.add_argument("--use_soft_logits", dest="use_soft_logits", type=bool, default=False)
    arg_parser.add_argument("--curriculum_learning", dest="curriculum_learning", type=bool, default=False)

    args = arg_parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    hparams = PARAMS_MAP[args.model]
    hparams["gpu_ids"] = list(range(len(args.gpu_ids.split(","))))
    hparams["root_dir"] = args.root_dir
    hparams["data_dir"] = args.data_dir
    hparams["bert_pretrained_dir"] = args.bert_pretrained_dir
    hparams["bert_pretrained"] = args.bert_pretrained
    hparams["bert_checkpoint_path"] = args.bert_checkpoint_path
    hparams["model_type"] = args.model
    hparams["task_name"] = args.task_name
    hparams["task_type"] = args.task_type
    hparams["training_type"] = args.training_type
    hparams["use_batch_negative"] = args.use_batch_negative
    hparams["use_soft_logits"] = args.use_soft_logits
    hparams["curriculum_learning"] = args.curriculum_learning

    set_random_seed(hparams['random_seed'])

    if len(args.electra_gen_config) > 0:
        hparams["electra_gen_config"] = args.electra_gen_config
        hparams["electra_gen_ckpt_path"] = "%s-pytorch_model.bin" % args.electra_gen_config

    # Multi-task types (ins, del, mod)
    multi_task_types = args.multi_task_type.split(",") if args.multi_task_type != '' else []

    for mt_type in multi_task_types:
        hparams.update(MULTI_TASK_TYPE_MAP[mt_type.strip()])

    hparams.update(DATASET_MAP[args.task_name])
    hparams.update(PRETRAINED_MODEL_MAP[args.bert_pretrained.split("-")[0]])

    if args.evaluate:
        evaluate_model(args, hparams)
    else:
        recall_list, model_path = train_model(args, hparams)
