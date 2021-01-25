import os
import torch
import pickle
import random
import copy

from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

from models.bert import tokenization_bert
from data.ubuntu_corpus_v1.ubuntu_data_utils import InputExamples


class ContrastiveResponseSelectionDataset(Dataset):
    """
    A full representation of VisDial v1.0 (train/val/test) dataset. According
    to the appropriate split, it returns dictionary of question, image,
    history, ground truth answer, answer options, dense annotations etc.
    """

    def __init__(
            self,
            hparams,
            split: str = "",
            data=None
    ):
        super().__init__()

        self.hparams = hparams
        self.split = split

        # read pkls -> Input Examples
        self.input_examples = []
        utterance_len_dict = dict()
        if data is None:
            data_path = os.path.join(hparams.data_dir, "%s_%s.pkl" % (hparams.task_name, split))
            with open(data_path, "rb") as pkl_handle:
                while True:
                    try:
                        example = pickle.load(pkl_handle)
                        num_examples = len(example.utterances) if len(example.utterances) < 10 else 10
                        try:
                            utterance_len_dict[str(num_examples)] += 1
                        except KeyError:
                            utterance_len_dict[str(num_examples)] = 1

                        if self.hparams.do_shuffle_ressel:
                            random.shuffle(example.utterances)

                        self.input_examples.append(example)

                        if len(self.input_examples) % 100000 == 0:
                            print("%d examples has been loaded!" % len(self.input_examples))
                            if self.hparams.pca_visualization:
                                break
                    except EOFError:
                        break
        else:
            bert_pretrained_dir = os.path.join(self.hparams.bert_pretrained_dir, self.hparams.bert_pretrained)
            self._bert_tokenizer = tokenization_bert.BertTokenizer(
                vocab_file=os.path.join(bert_pretrained_dir, "%s-vocab.txt" % self.hparams.bert_pretrained))
            contexts, responses = data
            for context, response in zip(contexts, responses):
                dialog_len = [len(x) for x in context]
                input_example = InputExamples(
                    utterances=context, response=response, label=1,
                    seq_lengths=(dialog_len, len(response)))
                self.input_examples.append(input_example)

            random.seed(self.hparams.random_seed)
            self.num_input_examples = len(self.input_examples)
            print("total %s examples" % split, self.num_input_examples)

        print(utterance_len_dict)
        integrated_input_examples = []
        if split == 'train':
            for i in range(0, len(self.input_examples), 2):  # training set 1:1
                integrated_input_examples.append((self.input_examples[i], self.input_examples[i + 1]))
        elif split in ('dev', 'test'):
            for i in range(0, len(self.input_examples), 10):  # dev/test set 1:10
                batch_data = tuple([self.input_examples[j] for j in range(i, i + 10)])
                integrated_input_examples.append(batch_data)
        else:  # dump logits
            integrated_input_examples = [(x,) for x in self.input_examples]
        self.input_examples = integrated_input_examples
        random.seed(self.hparams.random_seed)
        self.num_input_examples = len(self.input_examples)
        print("total %s examples" % split, self.num_input_examples)

        bert_pretrained_dir = os.path.join(self.hparams.bert_pretrained_dir, self.hparams.bert_pretrained)
        print(bert_pretrained_dir)
        self._bert_tokenizer = tokenization_bert.BertTokenizer(
            vocab_file=os.path.join(bert_pretrained_dir, "%s-vocab.txt" % self.hparams.bert_pretrained))

        self.del_placeholder = '[unused0]'

        # End of Turn Token
        if self.hparams.do_eot:
            self._bert_tokenizer.add_tokens(["[EOT]"])
        if self.hparams.do_sent_insertion:
            self._bert_tokenizer.add_tokens(["[INS]"])
        if self.hparams.do_sent_deletion:
            self._bert_tokenizer.add_tokens(["[DEL]"])
        if self.hparams.do_sent_search:
            self._bert_tokenizer.add_tokens(["[SRCH]"])

    def __len__(self):
        return len(self.input_examples)

    def __getitem__(self, index):
        # Get Input Examples
        """
        InputExamples
          self.utterances = utterances
          self.response = response
          self.label
        """
        if self.split == 'train':
            positive_example, negative_example = self.input_examples[index]
            pos_response_aug, neg_response_aug = self._nlp_augment(positive_example.response), self._nlp_augment(negative_example.response)
            all_responses = [positive_example.response, pos_response_aug, negative_example.response, neg_response_aug]
            dist_matrix = self._jaccard_distance_batch(all_responses)
            positive_example_aug, negative_example_aug = copy.deepcopy(positive_example), copy.deepcopy(negative_example)
            positive_example_aug.response = pos_response_aug
            positive_example_aug.response_len = len(pos_response_aug)
            negative_example_aug.response = neg_response_aug
            negative_example_aug.response_len = len(neg_response_aug)
            positive_example.soft_logits = [dist_matrix[0][1], dist_matrix[0][2], dist_matrix[0][3]]
            positive_example_aug.soft_logits = [dist_matrix[1][0], dist_matrix[1][2], dist_matrix[1][3]]
            negative_example.soft_logits = [dist_matrix[2][3], dist_matrix[2][0], dist_matrix[2][1]]
            negative_example_aug.soft_logits = [dist_matrix[3][2], dist_matrix[3][0], dist_matrix[3][1]]
            positive_feature = self._example_to_feature(index, positive_example)
            negative_feature = self._example_to_feature(index, negative_example)
            positive_feature_aug = self._example_to_feature(index, positive_example_aug)
            negative_feature_aug = self._example_to_feature(index, negative_example_aug)
            features = {'original': (positive_feature, negative_feature),
                        'augment': (positive_feature_aug, negative_feature_aug)}

            if self.split == 'train' and self.hparams.do_rank_loss:
                retrieve_response = positive_example.retrieve
                retrieve_example = copy.deepcopy(positive_example)
                retrieve_example.response = retrieve_response
                retrieve_example.response_len = len(retrieve_response)
                retrieve_feature = self._example_to_feature(index, retrieve_example)
                features['retrieve'] = retrieve_feature

        else:
            features = [self._example_to_feature(index, example) for example in self.input_examples[index]]

        return features

    def _nlp_augment(self, token_list, do_del=True, reorder=False):
        new_token_list = []
        if do_del:
            for i in range(len(token_list)):
                p = random.random()
                if p < 0.1:
                    new_token_list.append(self.del_placeholder)
                elif p < 0.2:
                    pass
                else:
                    new_token_list.append(token_list[i])
                if len(new_token_list) == 0:
                    new_token_list = token_list
        if reorder:
            raise NotImplementedError
        return new_token_list

    @staticmethod
    def _jaccard_distance_batch(uttrs):
        f_jaccard = lambda x, y: len(set(x) & set(y)) / len(set(x) | set(y))
        matrix = [[f_jaccard(uttrs[i], uttrs[j]) for j in range(len(uttrs))] for i in range(len(uttrs))]
        return matrix

    def _example_to_feature(self, index, example):
        current_feature = dict()
        anno_sent, segment_ids, attention_mask, eot_pos = self._annotate_sentence(example)
        current_feature["res_sel"] = dict()
        current_feature["res_sel"]["anno_sent"] = torch.tensor(anno_sent).long()
        current_feature["res_sel"]["segment_ids"] = torch.tensor(segment_ids).long()
        current_feature["res_sel"]["attention_mask"] = torch.tensor(attention_mask).long()
        current_feature["res_sel"]["eot_pos"] = torch.tensor(eot_pos).long()
        current_feature["res_sel"]["label"] = torch.tensor(example.label).float()
        if hasattr(example, 'soft_logits'):
            current_feature["res_sel"]["soft_logits"] = torch.tensor(example.soft_logits).float()

        # # when the response is the ground truth, append it to utterances.
        # if int(example.label) == 1:
        #     example.utterances.append(example.response)
        #
        # if len(example.utterances) == 1 and self.split == "train":
        #     return self._single_turn_processing(current_feature)  # why here?
        #
        # if self.hparams.do_sent_insertion and (self.split == "train" or self.hparams.pca_visualization):
        #     anno_sent, segment_ids, attention_mask, ins_pos, target_idx = self._insertion_annotate_sentence(example)
        #     current_feature["ins"] = dict()
        #     current_feature["ins"]["anno_sent"] = torch.tensor(anno_sent).long()
        #     current_feature["ins"]["segment_ids"] = torch.tensor(segment_ids).long()
        #     current_feature["ins"]["attention_mask"] = torch.tensor(attention_mask).long()
        #     current_feature["ins"]["ins_pos"] = torch.tensor(ins_pos).long()
        #     current_feature["ins"]["label"] = torch.tensor(target_idx).long()
        #
        # if self.hparams.do_sent_deletion and (self.split == "train" or self.hparams.pca_visualization):
        #     while True:
        #         target_idx = random.sample(list(range(self.num_input_examples)), 1)[0]
        #         target_example = self.input_examples[target_idx]
        #         if target_idx != index and len(target_example.utterances) > 2:
        #             break
        #     anno_sent, segment_ids, attention_mask, del_pos, target_idx = self._deletion_annotate_sentence(example, target_example)
        #     current_feature["del"] = dict()
        #     current_feature["del"]["anno_sent"] = torch.tensor(anno_sent).long()
        #     current_feature["del"]["segment_ids"] = torch.tensor(segment_ids).long()
        #     current_feature["del"]["attention_mask"] = torch.tensor(attention_mask).long()
        #     current_feature["del"]["del_pos"] = torch.tensor(del_pos).long()
        #     current_feature["del"]["label"] = torch.tensor(target_idx).long()
        #
        # if self.hparams.do_sent_search and (self.split == "train" or self.hparams.pca_visualization):
        #     anno_sent, segment_ids, attention_mask, srch_pos, target_idx = self._search_annotate_sentence(example)
        #     current_feature["srch"] = dict()
        #     current_feature["srch"]["anno_sent"] = torch.tensor(anno_sent).long()
        #     current_feature["srch"]["segment_ids"] = torch.tensor(segment_ids).long()
        #     current_feature["srch"]["attention_mask"] = torch.tensor(attention_mask).long()
        #     current_feature["srch"]["srch_pos"] = torch.tensor(srch_pos).long()
        #     current_feature["srch"]["label"] = torch.tensor(target_idx).long()

        return current_feature

    def _single_turn_processing(self, featrue: dict):
        max_seq_len = self.hparams.max_sequence_len
        if self.hparams.do_sent_insertion:
            featrue["ins"] = dict()
            featrue["ins"]["anno_sent"] = torch.tensor([0] * max_seq_len).long()
            featrue["ins"]["segment_ids"] = torch.tensor([0] * max_seq_len).long()
            featrue["ins"]["attention_mask"] = torch.tensor([0] * max_seq_len).long()
            featrue["ins"]["ins_pos"] = torch.tensor([0] * max_seq_len).long()
            featrue["ins"]["label"] = torch.tensor(-1).long()

        if self.hparams.do_sent_deletion:
            featrue["del"] = dict()
            featrue["del"]["anno_sent"] = torch.tensor([0] * max_seq_len).long()
            featrue["del"]["segment_ids"] = torch.tensor([0] * max_seq_len).long()
            featrue["del"]["attention_mask"] = torch.tensor([0] * max_seq_len).long()
            featrue["del"]["del_pos"] = torch.tensor([0] * max_seq_len).long()
            featrue["del"]["label"] = torch.tensor(-1).long()

        if self.hparams.do_sent_search:
            featrue["srch"] = dict()
            featrue["srch"]["anno_sent"] = torch.tensor([0] * max_seq_len).long()
            featrue["srch"]["segment_ids"] = torch.tensor([0] * max_seq_len).long()
            featrue["srch"]["attention_mask"] = torch.tensor([0] * max_seq_len).long()
            featrue["srch"]["srch_pos"] = torch.tensor([0] * max_seq_len).long()
            featrue["srch"]["label"] = torch.tensor(-1).long()

        return featrue

    def _search_annotate_sentence(self, example):
        """
            Search
        """
        max_utt_len = self.hparams.max_utt_len

        num_utterances = len(example.utterances)

        if num_utterances > max_utt_len:
            max_dialog_len_idx = random.sample(list(range(num_utterances - max_utt_len)), 1)[0]
            example.utterances = example.utterances[max_dialog_len_idx:max_dialog_len_idx + max_utt_len]
            num_utterances = len(example.utterances)

        utt_len = 3  # cls sep sep
        for utt_id, utt in enumerate(example.utterances):
            if len(utt) > int(self.hparams.max_sequence_len / 4):
                example.utterances[utt_id] = utt[:int(self.hparams.max_sequence_len / 4)]
            utt_len += len(utt) + 2  # srch, eot
            if utt_len > self.hparams.max_sequence_len:
                example.utterances = example.utterances[:utt_id]
                num_utterances = len(example.utterances)
                break

        target = example.utterances.pop() + ["[EOT]"]
        num_utterances -= 1

        random_utt_idx = list(range(num_utterances))
        random.shuffle(random_utt_idx)

        dialog_context = []
        target_idx = 0
        target_left = 0
        for i, random_id in enumerate(random_utt_idx):
            if random_id == num_utterances - 1:
                target_idx = i
                target_left = len(dialog_context)
            dialog_context.extend(["[SRCH]"] + example.utterances[random_id] + ["[EOT]"])

        target_right = len(dialog_context) - target_left
        dialog_context, target, target_idx = self._insert_max_len_trim_seq(dialog_context, target, target_idx,
                                                                           (target_left, target_right))

        # dialog context
        dialog_context = ["[CLS]"] + dialog_context + ["[SEP]"]
        segment_ids = [0] * len(dialog_context)
        attention_mask = [1] * len(dialog_context)

        target += ["[SEP]"]
        segment_ids.extend([1] * len(target))  # same utterance
        attention_mask.extend([1] * len(target))

        dialog_target = dialog_context + target

        while len(dialog_target) < self.hparams.max_sequence_len:
            dialog_target.append("[PAD]")
            segment_ids.append(0)
            attention_mask.append(0)

        srch_pos = []
        srch_cnt = 0
        for tok_idx, tok in enumerate(dialog_target):
            if tok == "[SRCH]":
                srch_pos.append(1)
                srch_cnt += 1
            else:
                srch_pos.append(0)

        assert len(dialog_target) == len(segment_ids) == len(attention_mask)
        assert len(dialog_target) <= self.hparams.max_sequence_len

        anno_sent = self._bert_tokenizer.convert_tokens_to_ids(dialog_target)

        return anno_sent, segment_ids, attention_mask, srch_pos, target_idx

    def _deletion_annotate_sentence(self, curr_example, target_example):
        max_utt_len = self.hparams.max_utt_len - 1

        target_sentence = random.sample(target_example.utterances, 1)[0]

        # TODO: current example
        # current example -> deletion is included
        num_utterances = len(curr_example.utterances)
        if num_utterances > max_utt_len:
            max_dialog_len_idx = random.sample(list(range(num_utterances - max_utt_len)), 1)[0]
            curr_example.utterances = curr_example.utterances[max_dialog_len_idx:max_dialog_len_idx + max_utt_len]
            num_utterances = max_utt_len

        for utt_i, utt in enumerate(curr_example.utterances):
            if len(utt) > int(self.hparams.max_sequence_len / 4):
                curr_example.utterances[utt_i] = utt[:int(self.hparams.max_sequence_len / 4)]

        curr_dialog_context = []
        delete_idx = random.sample(list(range(num_utterances)), 1)[0]

        delete_left = 0
        for utt_i, utt in enumerate(curr_example.utterances):
            if utt_i == delete_idx:
                delete_left = len(curr_dialog_context)
                curr_dialog_context.extend(["[DEL]"] + target_sentence + ["[EOT]"])
                if len(curr_example.utterances) > max_utt_len:
                    curr_example.utterances.pop()  # remove the last utterance
            curr_dialog_context.extend(["[DEL]"] + utt + ["[EOT]"])

        delete_right = len(curr_dialog_context) - delete_left

        target_dialog_context = []
        dialog_context, target_context, target_idx = \
            self._delete_max_len_trim_seq(curr_dialog_context, target_dialog_context, delete_idx,
                                          (delete_left, delete_right))

        # dialog context
        dialog_context = ["[CLS]"] + dialog_context + ["[SEP]"]
        segment_ids = [0] * len(dialog_context)
        attention_mask = [1] * len(dialog_context)

        dialog_target = dialog_context

        while len(dialog_target) < self.hparams.max_sequence_len:
            dialog_target.append("[PAD]")
            segment_ids.append(0)
            attention_mask.append(0)

        del_pos = []
        del_cnt = 0
        for tok_idx, tok in enumerate(dialog_target):
            if tok == "[DEL]":
                del_pos.append(1)
                del_cnt += 1
            else:
                del_pos.append(0)

        assert len(dialog_target) == len(segment_ids) == len(attention_mask)
        assert len(dialog_target) <= self.hparams.max_sequence_len

        anno_sent = self._bert_tokenizer.convert_tokens_to_ids(dialog_target)

        return anno_sent, segment_ids, attention_mask, del_pos, target_idx

    def _insertion_annotate_sentence(self, example):
        max_utt_len = self.hparams.max_utt_len

        num_utterances = len(example.utterances)

        if num_utterances > max_utt_len:
            max_dialog_len_idx = random.sample(list(range(num_utterances - max_utt_len)), 1)[0]
            example.utterances = example.utterances[max_dialog_len_idx:max_dialog_len_idx + max_utt_len]
            num_utterances = len(example.utterances)

        for utt_i, utt in enumerate(example.utterances):
            if len(utt) > int(self.hparams.max_sequence_len / 4):
                example.utterances[utt_i] = utt[:int(self.hparams.max_sequence_len / 4)]

        target = []
        dialog_context = ["[INS]"]
        target_idx = random.sample(list(range(num_utterances)), 1)[0]

        target_left, target_right = 0, 0
        for utt_i, utt in enumerate(example.utterances):
            if target_idx == utt_i:
                target_left = len(dialog_context) - 1
                target = utt + ["[EOT]"]
                continue
            dialog_context.extend(utt + ["[EOT]"] + ["[INS]"])

        target_right = len(dialog_context) - target_left
        dialog_context, target, target_idx = self._insert_max_len_trim_seq(dialog_context, target, target_idx,
                                                                           (target_left, target_right))

        # dialog context
        dialog_context = ["[CLS]"] + dialog_context + ["[SEP]"]
        segment_ids = [0] * len(dialog_context)
        attention_mask = [1] * len(dialog_context)

        target += ["[SEP]"]
        segment_ids.extend([1] * len(target))  # same utterance
        attention_mask.extend([1] * len(target))

        dialog_target = dialog_context + target

        while len(dialog_target) < self.hparams.max_sequence_len:
            dialog_target.append("[PAD]")
            segment_ids.append(0)
            attention_mask.append(0)

        ins_pos = []
        ins_cnt = 0
        for tok_idx, tok in enumerate(dialog_target):
            if tok == "[INS]":
                ins_pos.append(1)
                ins_cnt += 1
            else:
                ins_pos.append(0)
        assert len(dialog_target) == len(segment_ids) == len(attention_mask)
        assert len(dialog_target) <= self.hparams.max_sequence_len

        anno_sent = self._bert_tokenizer.convert_tokens_to_ids(dialog_target)

        return anno_sent, segment_ids, attention_mask, ins_pos, target_idx

    def _annotate_sentence(self, example):

        dialog_context = []
        if self.hparams.do_eot:
            for utt in example.utterances:
                dialog_context.extend(utt + ["[EOT]"])
        else:
            for utt in example.utterances:
                dialog_context.extend(utt)
        response = example.response + ["[EOT]"]
        dialog_context, response = self._max_len_trim_seq(dialog_context, response)

        # dialog context
        dialog_context = ["[CLS]"] + dialog_context + ["[SEP]"]
        segment_ids = [0] * len(dialog_context)
        attention_mask = [1] * len(dialog_context)

        response = response + ["[SEP]"]
        segment_ids.extend([1] * len(response))
        attention_mask.extend([1] * len(response))

        dialog_response = dialog_context + response

        while len(dialog_response) < self.hparams.max_sequence_len:
            dialog_response.append("[PAD]")
            segment_ids.append(0)
            attention_mask.append(0)

        eot_pos = []
        for tok_idx, tok in enumerate(dialog_response):
            if tok == "[EOT]":
                eot_pos.append(1)
            else:
                eot_pos.append(0)

        assert len(dialog_response) == len(segment_ids) == len(attention_mask)
        anno_sent = self._bert_tokenizer.convert_tokens_to_ids(dialog_response)
        assert len(dialog_response) <= self.hparams.max_sequence_len

        return anno_sent, segment_ids, attention_mask, eot_pos

    def _delete_max_len_trim_seq(self, curr_dialog_context, target_dialog_context, target_idx, lengths):
        delete_left, delete_right = lengths

        while len(curr_dialog_context) + len(target_dialog_context) > self.hparams.max_sequence_len - 3:
            if len(curr_dialog_context) > len(target_dialog_context):
                if delete_left > delete_right:
                    if curr_dialog_context[0] in ["[DEL]"]:
                        target_idx -= 1
                    delete_left -= 1
                    curr_dialog_context.pop(0)  # from the left
                else:
                    delete_right -= 1
                    curr_dialog_context.pop()  # from the right
            else:
                target_dialog_context.pop(0)

        return curr_dialog_context, target_dialog_context, target_idx

    def _insert_max_len_trim_seq(self, dialog_context, target, target_idx, lengths):

        target_left, target_right = lengths
        # [CLS] [SEP] [EOT] [SEP]
        while len(dialog_context) + len(target) > self.hparams.max_sequence_len - 3:
            if len(dialog_context) > len(target):
                if target_left > target_right:
                    if dialog_context[0] in ["[INS]"]:
                        target_idx -= 1
                    target_left -= 1
                    dialog_context.pop(0)  # from the left
                else:
                    target_right -= 1
                    dialog_context.pop()  # from the right
            else:
                target.pop()

        return dialog_context, target, target_idx

    def _max_len_trim_seq(self, dialog_context, response):

        while len(dialog_context) + len(response) > self.hparams.max_sequence_len - 3:
            if len(dialog_context) > len(response):
                dialog_context.pop(0)  # from the front
            else:
                response.pop()

        # while len(dialog_context) > 446:
        #     dialog_context.pop(0)
        # while len(response) > 63:
        #     response.pop()

        return dialog_context, response

    @staticmethod
    def collate_fn(batch):
        if isinstance(batch[0], dict):  # train
            feature_dict = {}
            for key in batch[0]:
                feature_dict[key] = []
            for example in batch:
                for group in ('original', 'augment'):
                    pos_example, neg_example = example[group]
                    feature_dict[group].extend([pos_example, neg_example])
                    # group_example_list = feature_dict.get(group, [])
                    # group_example_list.extend([pos_example, neg_example])
                    # feature_dict[group] = group_example_list
                if 'retrieve' in feature_dict:
                    feature_dict['retrieve'].append(example['retrieve'])
            ret = {}
            for example_group in feature_dict:
                ret[example_group] = default_collate(feature_dict[example_group])
        else:  # dev & test
            ret = default_collate(batch[0])
        return ret
