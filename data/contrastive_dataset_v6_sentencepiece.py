import os
import torch
import pickle
import random
import copy

from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
import sentencepiece as spm

from data.ubuntu_corpus_v1.ubuntu_data_utils import InputExamples
import global_variables


class ContrastiveResponseSelectionDatasetSentencepiece(Dataset):
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
        if hparams.task_name == 'ubuntu':
            from contrastive.eda.en_eda.code.eda import eda
            self.eda = eda
        else:
            from contrastive.eda.zh_eda.code.eda import eda
            self.eda = eda
        self.mask_token = False
        self.sp_tokenizer = spm.SentencePieceProcessor(model_file='data/ubuntu_corpus_v1/ubuntu_sp_0.9995.model')

        # read pkls -> Input Examples
        self.input_examples = []
        utterance_len_dict = dict()
        dialogues = []
        with open(os.path.join(hparams.data_dir, f"{split}.txt")) as fr:
            for line in fr:
                label, dialogue = line.strip().split('\t', 1)
                turns = dialogue.split('\t')
                context, response = turns[:-1], turns[-1]
                dialogues.append((label, context, response))
        for label, context, response in dialogues:
            context = [self.sp_tokenizer.encode(x, out_type=str) for x in context]
            response = self.sp_tokenizer.encode(response, out_type=str)
            dialog_len = [len(x) for x in context]
            input_example = InputExamples(
                utterances=context, response=response, label=int(label),
                seq_lengths=(dialog_len, len(response)))
            self.input_examples.append(input_example)

        random.seed(self.hparams.random_seed)
        self.num_input_examples = len(self.input_examples)
        print("total %s examples" % split, self.num_input_examples)
        print(utterance_len_dict)
        integrated_input_examples = []
        if split == 'train':
            for i in range(0, len(self.input_examples), 2):  # training set 1:1
                if len(self.input_examples[i].response) == 0 or len(self.input_examples[i + 1].response) == 0:
                    continue
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
            p_context_augment = global_variables.epoch / 10
            positive_example.label2 = 0
            negative_example.label2 = 2

            pos_response_aug, neg_response_aug = self._nlp_augment(positive_example.response), self._nlp_augment(negative_example.response)
            # all_responses = [positive_example.response, pos_response_aug, negative_example.response, neg_response_aug]
            # dist_matrix = self._edit_distance_similarity_batch(all_responses)
            positive_example_aug, negative_example_aug = copy.deepcopy(positive_example), copy.deepcopy(negative_example)
            positive_example_aug.response = pos_response_aug
            positive_example_aug.response_len = len(pos_response_aug)
            negative_example_aug.response = neg_response_aug
            negative_example_aug.response_len = len(neg_response_aug)
            # positive_example.soft_logits = [dist_matrix[0][1], dist_matrix[0][2], dist_matrix[0][3]]
            # positive_example_aug.soft_logits = [dist_matrix[1][0], dist_matrix[1][2], dist_matrix[1][3]]
            # negative_example.soft_logits = [dist_matrix[2][3], dist_matrix[2][0], dist_matrix[2][1]]
            # negative_example_aug.soft_logits = [dist_matrix[3][2], dist_matrix[3][0], dist_matrix[3][1]]
            positive_feature = self._example_to_feature(index, positive_example)
            negative_feature = self._example_to_feature(index, negative_example)
            positive_feature_aug = self._example_to_feature(index, positive_example_aug)
            negative_feature_aug = self._example_to_feature(index, negative_example_aug)

            features = {'original': (positive_feature, negative_feature),
                        'augment': (positive_feature_aug, negative_feature_aug)}

            # extra contrastive data
            contras_idx = random.randint(0, len(self.input_examples) // 2 - 1)
            contrastive_positive_example = self.input_examples[2 * contras_idx][0]
            dialogue = contrastive_positive_example.utterances + [contrastive_positive_example.response]
            n_context_turns = len(dialogue)
            if n_context_turns <= 2:
                st_turn, ed_turn = 0, n_context_turns - 2
                positive_response = dialogue[-1]
                less_positive_response = dialogue[0]
                negative_response = random.sample(self.input_examples, 1)[0][0].response
            else:
                n_new_context_turns = random.randint(1, n_context_turns - 2)
                st_turn = random.randint(0, n_context_turns - n_new_context_turns - 1)
                ed_turn = st_turn + n_new_context_turns - 1
                response_candidates = (set(range(0, st_turn)) | set(range(ed_turn + 2, n_context_turns)))
                positive_response = dialogue[ed_turn + 1]
                less_positive_response = dialogue[random.sample(response_candidates, 1)[0]]
                negative_response = random.sample(self.input_examples, 1)[0][0].response
            positive_example_contras = InputExamples(
                utterances=dialogue[st_turn:ed_turn + 1], response=positive_response, label=1,
                seq_lengths=([len(x) for x in dialogue[st_turn:ed_turn + 1]], len(positive_response)), label2=0
            )
            less_positive_example_contras = InputExamples(
                utterances=dialogue[st_turn:ed_turn + 1], response=less_positive_response, label=0,
                seq_lengths=([len(x) for x in dialogue[st_turn:ed_turn + 1]], len(less_positive_response)), label2=1
            )
            negative_example_contras = InputExamples(
                utterances=dialogue[st_turn:ed_turn + 1], response=negative_response, label=0,
                seq_lengths=([len(x) for x in dialogue[st_turn:ed_turn + 1]], len(negative_response)), label2=2
            )
            positive_response_aug = self._nlp_augment(positive_response)
            less_positive_response_aug = self._nlp_augment(less_positive_response)
            negative_response_aug = self._nlp_augment(negative_response)
            positive_example_contras_aug = InputExamples(
                utterances=dialogue[st_turn:ed_turn + 1], response=positive_response_aug, label=-1,
                seq_lengths=([len(x) for x in dialogue[st_turn:ed_turn + 1]], len(positive_response_aug)), label2=0
            )
            less_positive_example_contras_aug = InputExamples(
                utterances=dialogue[st_turn:ed_turn + 1], response=less_positive_response_aug, label=-1,
                seq_lengths=([len(x) for x in dialogue[st_turn:ed_turn + 1]], len(less_positive_response_aug)), label2=1
            )
            negative_example_contras_aug = InputExamples(
                utterances=dialogue[st_turn:ed_turn + 1], response=negative_response_aug, label=-1,
                seq_lengths=([len(x) for x in dialogue[st_turn:ed_turn + 1]], len(negative_response_aug)), label2=2
            )
            positive_feature_contras = self._example_to_feature(index, positive_example_contras)
            less_positive_feature_contras = self._example_to_feature(index, less_positive_example_contras)
            negative_feature_contras = self._example_to_feature(index, negative_example_contras)
            positive_feature_contras_aug = self._example_to_feature(index, positive_example_contras_aug)
            less_positive_feature_contras_aug = self._example_to_feature(index, less_positive_example_contras_aug)
            negative_feature_contras_aug = self._example_to_feature(index, negative_example_contras_aug)

            features.update({
                'contras': (positive_feature_contras, negative_feature_contras),
                'contras_aug': (positive_feature_contras_aug, negative_feature_contras_aug),
                'sample': (less_positive_feature_contras, less_positive_feature_contras_aug),
            })

            if self.split == 'train' and (self.hparams.do_sent_insertion or self.hparams.do_sent_deletion or self.hparams.do_sent_search):
                extra_example = copy.deepcopy(positive_example)
                extra_example.utterances.append(extra_example.response)
                current_feature = dict()
                if self.hparams.do_sent_insertion:
                    anno_sent, segment_ids, attention_mask, ins_pos, target_idx = self._insertion_annotate_sentence(extra_example)
                    current_feature["ins"] = dict()
                    current_feature["ins"]["anno_sent"] = torch.tensor(anno_sent).long()
                    current_feature["ins"]["segment_ids"] = torch.tensor(segment_ids).long()
                    current_feature["ins"]["attention_mask"] = torch.tensor(attention_mask).long()
                    current_feature["ins"]["ins_pos"] = torch.tensor(ins_pos).long()
                    current_feature["ins"]["label"] = torch.tensor(target_idx).long()
                    features.update({'extra': current_feature})
                if self.hparams.do_sent_deletion:
                    while True:
                        target_idx = random.sample(list(range(self.num_input_examples)), 1)[0]
                        target_example = self.input_examples[target_idx][0]
                        if target_idx != index and len(target_example.utterances) > 2:
                            break
                    anno_sent, segment_ids, attention_mask, del_pos, target_idx = self._deletion_annotate_sentence(
                        extra_example, target_example)
                    current_feature["del"] = dict()
                    current_feature["del"]["anno_sent"] = torch.tensor(anno_sent).long()
                    current_feature["del"]["segment_ids"] = torch.tensor(segment_ids).long()
                    current_feature["del"]["attention_mask"] = torch.tensor(attention_mask).long()
                    current_feature["del"]["del_pos"] = torch.tensor(del_pos).long()
                    current_feature["del"]["label"] = torch.tensor(target_idx).long()
                if self.hparams.do_sent_search:
                    anno_sent, segment_ids, attention_mask, srch_pos, target_idx = self._search_annotate_sentence(extra_example)
                    current_feature["srch"] = dict()
                    current_feature["srch"]["anno_sent"] = torch.tensor(anno_sent).long()
                    current_feature["srch"]["segment_ids"] = torch.tensor(segment_ids).long()
                    current_feature["srch"]["attention_mask"] = torch.tensor(attention_mask).long()
                    current_feature["srch"]["srch_pos"] = torch.tensor(srch_pos).long()
                    current_feature["srch"]["label"] = torch.tensor(target_idx).long()

        else:
            features = [self._example_to_feature(index, example) for example in self.input_examples[index]]

        return features

    def _nlp_augment(self, token_list, do_del=True, do_reorder=True):
        augment_alpha = 0.2#  * min(global_variables.epoch, 2)
        if self.hparams.task_name == 'ubuntu':
            text = ' '.join([x for x in token_list]).replace(' ##', '')
            new_text = self.eda(text, alpha_sr=augment_alpha, alpha_ri=augment_alpha, alpha_rs=augment_alpha)[0]
        else:
            text = ''.join(token_list)
            new_text = self.eda(text, alpha_sr=0, alpha_ri=augment_alpha, alpha_rs=augment_alpha)[0]
        # new_text = text
        new_token_list = self.sp_tokenizer.encode(new_text, out_type=str)
        return new_token_list

    @staticmethod
    def _jaccard_similarity(x, y):
        return len(set(x) & set(y)) / len(set(x) | set(y))

    @staticmethod
    def _edit_distance_similarity_batch(uttrs):
        def edit_distance(str1, str2):
            matrix = [[i + j for j in range(len(str2) + 1)] for i in range(len(str1) + 1)]
            for i in range(1, len(str1) + 1):
                for j in range(1, len(str2) + 1):
                    if str1[i - 1] == str2[j - 1]:
                        d = 0
                    else:
                        d = 1
                    matrix[i][j] = min(matrix[i - 1][j] + 1, matrix[i][j - 1] + 1, matrix[i - 1][j - 1] + d)
            return matrix[len(str1)][len(str2)]
        f_edit_similarity = lambda x, y: 1 / (edit_distance(x, y) + 1e-3)
        matrix = [[f_edit_similarity(uttrs[i], uttrs[j]) for j in range(len(uttrs))] for i in range(len(uttrs))]
        return matrix

    def _example_to_feature(self, index, example):
        current_feature = dict()
        anno_sent, segment_ids, attention_mask, eot_pos = self._annotate_sentence(example)
        current_feature["res_sel"] = dict()
        current_feature["res_sel"]["anno_sent"] = torch.tensor(anno_sent).long()
        current_feature["res_sel"]["segment_ids"] = torch.tensor(segment_ids).long()
        current_feature["res_sel"]["attention_mask"] = torch.tensor(attention_mask).long()
        current_feature["res_sel"]["eot_pos"] = torch.tensor(eot_pos).long()
        current_feature["res_sel"]["label"] = torch.tensor(example.label)
        if hasattr(example, 'label2'):
            current_feature["res_sel"]["label2"] = torch.tensor(example.label2)

        return current_feature

    def _annotate_sentence(self, example):

        dialog_context = []
        if self.hparams.do_eot:
            for utt in example.utterances:
                if self.mask_token:
                    utt = [tok if random.random() > 0.15 else '[MASK]' for tok in utt]
                dialog_context.extend(utt + ["[EOT]"])
        else:
            for utt in example.utterances:
                if self.mask_token:
                    utt = [tok if random.random() > 0.15 else '[MASK]' for tok in utt]
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
        anno_sent = self.sp_tokenizer.piece_to_id(dialog_response)
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
                    if dialog_context[0] in [self.ins_token]:
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
                for group in ('original', 'augment', 'contras', 'contras_aug', 'sample'):
                    if group not in example:
                        continue
                    pos_example, neg_example = example[group]
                    feature_dict[group].extend([pos_example, neg_example])
                    # group_example_list = feature_dict.get(group, [])
                    # group_example_list.extend([pos_example, neg_example])
                    # feature_dict[group] = group_example_list
                for group in ('retrieve', 'extra'):
                    if group in feature_dict:
                        feature_dict[group].append(example[group])
            ret = {}
            for example_group in feature_dict:
                ret[example_group] = default_collate(feature_dict[example_group])
        else:  # dev & test
            ret = default_collate(batch[0])
        return ret
