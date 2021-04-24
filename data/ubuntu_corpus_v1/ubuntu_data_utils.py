import os
import sys

sys.path.append(os.getcwd())
import pickle
from tqdm import tqdm, trange

from models.bert import tokenization_bert
from contrastive.cl_utils import ContrastiveUtils


class InputExamples(object):
    def __init__(self, utterances, response, label, seq_lengths, augments=None, retrieve=None, label2=-1):
        self.utterances = utterances
        self.response = response
        self.label = label

        self.dialog_len = seq_lengths[0]
        self.response_len = seq_lengths[1]
        self.augments = augments
        self.retrieve = retrieve
        self.label2 = label2


class UbuntuDataUtils(object):
    def __init__(self, txt_path, bert_pretrained_dir, bert_pretrained):
        # bert_tokenizer init
        self.txt_path = txt_path
        self._bert_tokenizer_init(bert_pretrained_dir, bert_pretrained)
        # self.contrastive_util = ContrastiveUtils(lang='en')

    def _bert_tokenizer_init(self, bert_pretrained_dir, bert_pretrained='bert-base-uncased'):

        self._bert_tokenizer = tokenization_bert.BertTokenizer(
            vocab_file=os.path.join(os.path.join(bert_pretrained_dir, bert_pretrained),
                                    "%s-vocab.txt" % bert_pretrained))
        print("BERT tokenizer init completes")

    def read_raw_file(self, data_type):
        print("Loading raw txt file...")

        ubuntu_path = self.txt_path % data_type  # train, dev, test
        with open(ubuntu_path, "r", encoding="utf8") as fr_handle:
            data = [line.strip() for line in fr_handle if len(line.strip()) > 0]
            print("(%s) total number of sentence : %d" % (data_type, len(data)))

        return data

    def make_examples_pkl(self, data, ubuntu_pkl_path, do_augment=True, do_retrieve=True):
        if do_augment:
            ubuntu_pkl_path = ubuntu_pkl_path[:-4] + '_aug.pkl'
        if do_retrieve:
            ubuntu_pkl_path = ubuntu_pkl_path[:-4] + '_retrieve.pkl'
        responses = []
        for dialog in tqdm(data):
            dialog_data = dialog.split("\t")
            responses.append(dialog_data[-1])
        responses_augs = self.contrastive_util.batch_back_translation_augmentation(responses)
        retrieve_response_lines = open('data/ubuntu_corpus_v1/retrieve.txt', 'r', encoding='utf-8').readlines()

        with open(ubuntu_pkl_path, "wb") as pkl_handle:
            for dialog_idx, dialog in enumerate(tqdm(data)):
                dialog_data = dialog.split("\t")
                label = dialog_data[0]
                utterances = []
                dialog_len = []

                for utt in dialog_data[1:-1]:
                    utt_tok = self._bert_tokenizer.tokenize(utt)
                    utterances.append(utt_tok)
                    dialog_len.append(len(utt_tok))
                response = self._bert_tokenizer.tokenize(dialog_data[-1])

                augments = None
                if do_augment:
                    response_aug1, response_aug2 = responses_augs[dialog_idx]
                    response_aug1 = self._bert_tokenizer.tokenize(response_aug1)
                    response_aug2 = self._bert_tokenizer.tokenize(response_aug2)
                    augments = response_aug1, response_aug2

                retrieve = None
                if do_retrieve:
                    retrieve_response = retrieve_response_lines[dialog_idx].strip()
                    retrieve = self._bert_tokenizer.tokenize(retrieve_response)

                pickle.dump(InputExamples(
                    utterances=utterances, response=response, label=int(label),
                    seq_lengths=(dialog_len, len(response)), augments=augments, retrieve=retrieve), pkl_handle)

        print(ubuntu_pkl_path, " save completes!")

    def make_example_pkl_retrieve(self):
        ubuntu_pkl_path = 'data/ubuntu_corpus_v1/ubuntu_train_aug.pkl'
        new_ubuntu_pkl_path = ubuntu_pkl_path[:-4] + '_retrieve.pkl'
        retrieve_responses = open('data/ubuntu_corpus_v1/retrieve_response.txt', 'r', encoding='utf-8').readlines()
        with open(new_ubuntu_pkl_path, 'wb') as fw:
            with open(ubuntu_pkl_path, 'rb') as fr:
                for i in trange(1000000):
                    example_idx = i // 2
                    example = pickle.load(fr)
                    retrieve_response = retrieve_responses[example_idx].strip()
                    retrieve = self._bert_tokenizer.tokenize(retrieve_response)
                    example.retrieve = retrieve
                    pickle.dump(example, fw)


if __name__ == '__main__':
    ubuntu_raw_path = "./data/ubuntu_corpus_v1/%s.txt"
    ubuntu_pkl_path = "./data/ubuntu_corpus_v1/ubuntu_%s.pkl"
    bert_pretrained = "bert-base-uncased"
    bert_pretrained_dir = "./resources"

    ubuntu_utils = UbuntuDataUtils(ubuntu_raw_path, bert_pretrained_dir, bert_pretrained)

    # # response seleciton fine-tuning pkl creation
    # for data_type in ["train", "valid", "test"]:
    #     data = ubuntu_utils.read_raw_file(data_type)
    #     ubuntu_utils.make_examples_pkl(data, ubuntu_pkl_path % data_type, do_augment=True)

    ubuntu_utils.make_example_pkl_retrieve()
