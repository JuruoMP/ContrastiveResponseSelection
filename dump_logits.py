import argparse
import collections
import os
import pickle as pkl

from tqdm import trange

from config.hparams import *
from evaluation import Evaluation
from contrastive.cl_evaluation import ContrastiveEvaluation
from post_train.post_training import PostTraining
from train import ResponseSelection

class InputExamples(object):
    def __init__(self, utterances, response, label, seq_lengths, augments=None, retrieve=None):
        self.utterances = utterances
        self.response = response
        self.label = label

        self.dialog_len = seq_lengths[0]
        self.response_len = seq_lengths[1]
        self.augments = augments
        self.retrieve = retrieve

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
    "post_training": PostTraining
}

EVAL_TYPE_MAP = {
    "fine_tuning": Evaluation,
    "contrastive": ContrastiveEvaluation
}

MULTI_TASK_TYPE_MAP = {
    "ins": INSERTION_PARAMS,
    "del": DELETION_PARAMS,
    "srch": SEARCH_PARAMS
}


def evaluate(hparams, context_list, response_list):
    test_data = (context_list, response_list)
    hparams = collections.namedtuple("HParams", sorted(hparams.keys()))(**hparams)
    model = EVAL_TYPE_MAP[args.training_type](hparams)
    logits = model.run_evaluate_with_data(args.evaluate, test_data)
    return logits


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description="Utterance Manipulation Strategy : Response Selection (PyTorch)")
    arg_parser.add_argument("--model", dest="model", type=str, default="bert_base", help="Model Name")
    arg_parser.add_argument("--task_name", dest="task_name", type=str, default="ubuntu", help="Task Name")
    arg_parser.add_argument("--task_type", dest="task_type", type=str,
                            default="response_selection",
                            help="response selection | sentence insertion")
    # bert-base-uncased, bert-post-uncased
    arg_parser.add_argument("--root_dir", dest="root_dir", type=str,
                            default="/data/taesunwhang/response_selection/",
                            help="model train logs, checkpoints")
    arg_parser.add_argument("--data_dir", dest="data_dir", type=str, default='data/ubuntu_corpus_v1',
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
    arg_parser.add_argument("--logits_path", dest="logits_path", type=str, default="",
                            help="file path of soft logits")
    arg_parser.add_argument("--training_type", dest="training_type", type=str, default="contrastive",
                            help="fine_tuning or post_training")
    arg_parser.add_argument("--multi_task_type", dest="multi_task_type", type=str, default="",
                            help="ins,del,srch")
    arg_parser.add_argument("--gpu_ids", dest="gpu_ids", type=str,
                            help="gpu_ids", default="0, 1")
    arg_parser.add_argument("--electra_gen_config", dest="electra_gen_config", type=str,
                            help="electra_gen_config", default="")  # electra-base-gen, electra-base-chinese-gen
    arg_parser.add_argument("--input", dest="input_path", type=str,
                            help="Input pkl path", default="")
    arg_parser.add_argument("--output", dest="output_path", type=str,
                            help="Output pkl path", default="")

    args = arg_parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    n_augment = 2

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
    hparams["logits_path"] = args.logits_path

    if len(args.electra_gen_config) > 0:
        hparams["electra_gen_config"] = args.electra_gen_config
        hparams["electra_gen_ckpt_path"] = "%s-pytorch_model.bin" % args.electra_gen_config

    # Multi-task types (ins, del, mod)
    multi_task_types = args.multi_task_type.split(",") if args.multi_task_type != '' else []

    for mt_type in multi_task_types:
        hparams.update(MULTI_TASK_TYPE_MAP[mt_type.strip()])

    hparams.update(DATASET_MAP[args.task_name])
    hparams.update(PRETRAINED_MODEL_MAP[args.bert_pretrained.split("-")[0]])
    hparams.update({'evaluate_data_type': 'train'})

    context_list, response_list = [], []
    all_examples = []
    with open(args.input_path, 'rb') as pkl_handler:
        while True:
            try:
                example = pkl.load(pkl_handler)
                context_list.append(example.utterances)
                response_list.append(example.response)
                assert len(example.augments) == n_augment
                for i in range(n_augment):
                    context_list.append(example.utterances)
                    response_list.append(example.augments[i])
                all_examples.append(example)
                if len(all_examples) % 100000 == 0:
                    print(f'Load {len(all_examples)} examples')
            except EOFError:
                break
    logits = evaluate(hparams, context_list, response_list)
    with open(args.output_path, 'wb') as pkl_handler:
        for i in trange(len(all_examples)):
            example = all_examples[i]
            example.soft_logits = logits[3 * i]
            example.aug_soft_logits = (logits[3 * i + 1], logits[3 * i + 2])
            pkl.dump(example, pkl_handler)
    print('Dump soft logits successfully')
