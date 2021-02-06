import os
import random
import torch
import torch.nn as nn
from torch.cuda import amp
import numpy as np

from models.contrastive_loss import NTXentLoss, DynamicNTXentLoss, SupConLoss
import global_variables
from models.bert_insertion import BertInsertion
from models.bert_deletion import BertDeletion
from models.bert_search import BertSearch


class SelfAttention(nn.Module):
    def __init__(self, d_hid, dropout=0.):
        super().__init__()
        self.scorer = nn.Linear(d_hid, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_seq, mask):
        batch_size, seq_len, feature_dim = input_seq.size()
        input_seq = self.dropout(input_seq)
        scores = self.scorer(input_seq.contiguous().view(-1, feature_dim)).view(batch_size, seq_len)
        scores = scores.masked_fill((1 - mask).bool(), -np.inf)
        scores = torch.softmax(scores, dim=1)
        context = scores.unsqueeze(2).expand_as(input_seq).mul(input_seq).sum(1)
        return context


class BertCls(nn.Module):
    def __init__(self, hparams):
        super(BertCls, self).__init__()
        self.hparams = hparams

        pretrained_config = hparams.pretrained_config.from_pretrained(
            os.path.join(self.hparams.bert_pretrained_dir, self.hparams.bert_pretrained,
                         "%s-config.json" % self.hparams.bert_pretrained),
        )
        pretrained_config.output_hidden_states = True
        self._model = hparams.pretrained_model.from_pretrained(
            os.path.join(self.hparams.bert_pretrained_dir, self.hparams.bert_pretrained,
                         self.hparams.bert_checkpoint_path),
            config=pretrained_config
        )

        if hparams.bert_freeze_layer > 0:
            freeze_layers = self._model.encoder.layer
            for param in freeze_layers:
                param.requires_grad = False

        num_new_tok = 0
        if self.hparams.model_type.startswith("bert_base") or self.hparams.model_type.startswith("electra_base"):
            if self.hparams.do_eot:
                num_new_tok += 1
            # bert_post already has [EOT]

        self._model.resize_token_embeddings(self._model.config.vocab_size + num_new_tok)  # [EOT]

        if self.hparams.do_sent_insertion:
            self._bert_insertion = BertInsertion(hparams, self._model)
        if self.hparams.do_sent_deletion:
            self._bert_deletion = BertDeletion(hparams, self._model)
        if self.hparams.do_sent_search:
            self._bert_search = BertSearch(hparams, self._model)

        self._classification = nn.Sequential(
            nn.Dropout(p=1 - self.hparams.dropout_keep_prob),
            nn.Linear(self.hparams.bert_hidden_dim, 1)
        )

        self._classification_attn = nn.Sequential(
            nn.Dropout(p=1 - self.hparams.dropout_keep_prob),
            nn.Linear(self.hparams.bert_hidden_dim * 2, 1)
        )

        self._classification_list = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(p=1 - self.hparams.dropout_keep_prob),
                nn.Linear(self.hparams.bert_hidden_dim, 1)
            ) for _ in range(4)
        ])

        self.self_attention = SelfAttention(pretrained_config.hidden_size)

        self._classification2 = nn.Sequential(
            nn.Dropout(p=1 - self.hparams.dropout_keep_prob),
            nn.Linear(self.hparams.bert_hidden_dim, 3)
        )

        self._projection = nn.Sequential(
            nn.Linear(self.hparams.bert_hidden_dim, self.hparams.projection_dim),
            nn.ReLU(),
            nn.Linear(self.hparams.projection_dim, self.hparams.projection_dim)
        )

        self._projection_sup = nn.Sequential(
            nn.Linear(self.hparams.bert_hidden_dim, self.hparams.projection_dim),
            nn.ReLU(),
            nn.Linear(self.hparams.projection_dim, self.hparams.projection_dim)
        )

        self._criterion = nn.BCEWithLogitsLoss(reduction='none')
        if self.hparams.use_batch_negative:
            self._nt_xent_criterion = NTXentLoss(temperature=0.5, use_cosine_similarity=True)
        self._nt_xent_criterion = DynamicNTXentLoss(temperature=0.5, use_cosine_similarity=True)
        self._sup_con_loss = SupConLoss(temperature=0.1)

    @amp.autocast()
    def forward(self, batch_data):

        bert_output_cache = {
            'original': None,
            'augment': None,
            'contras': None,
            'contras_aug': None,
            'sample': None,
            'extra': None,
        }
        def get_bert_output(name, return_hidden=False, return_layer=False):
            if bert_output_cache.get(name) is not None:
                cls_logits, hidden_states_cls, bert_outputs = bert_output_cache.get(name)
            else:
                batch = batch_data[name]
                bert_outputs, _, hidden_states = self._model(
                    batch["res_sel"]["anno_sent"],
                    token_type_ids=batch["res_sel"]["segment_ids"],
                    attention_mask=batch["res_sel"]["attention_mask"]
                )
                cls_logits = bert_outputs[:, 0, :]  # bs, bert_output_size
                hidden_states_cls = [x[:, 0, :] for x in hidden_states]
                bert_output_cache[name] = (cls_logits, hidden_states_cls, bert_outputs)
            ret = (cls_logits,)
            if return_hidden:
                ret = ret + (hidden_states_cls,)
            if return_layer:
                ret = ret + (bert_outputs,)
            return ret

        device = batch_data['original']['res_sel']['anno_sent'].device
        use_all_bert_output = False
        use_multi_layers = False
        if self.training:  # training
            original_response_selection, original_contrastive = True, self.hparams.do_contrastive
            new_response_selection, new_contrastive = False, False
            less_positive_res_sel = False
            three_class_classification = False
            supervised_contrastive = False
        else:  # evaluation
            original_response_selection, original_contrastive = True, False
            new_response_selection, new_contrastive = False, False
        logits = torch.Tensor([0]).to(device)
        res_sel_loss_list, contrastive_loss_list = [], []
        if original_response_selection or original_contrastive:
            cls_logits, all_cls_logits, last_layer_logits = get_bert_output('original', return_hidden=True, return_layer=True)
            if original_response_selection:
                if use_all_bert_output:
                    attn_vector = self.self_attention(last_layer_logits, batch_data['original']['res_sel']['attention_mask'])
                    state_vector = torch.cat((cls_logits, attn_vector), dim=-1)
                    logits = self._classification_attn(state_vector)
                else:
                    logits = self._classification(cls_logits)  # bs, 1
                logits = logits.squeeze(-1)
                res_sel_losses = self._criterion(logits, batch_data['original']["res_sel"]["label"].float())
                mask = batch_data['original']["res_sel"]["label"] == -1
                res_sel_loss = res_sel_losses.masked_fill(mask, 0).mean()
                res_sel_loss_list.append(res_sel_loss)
                ####################### multi layer cls prediction ########################
                if use_multi_layers:
                    chosen_layer_cls_logits = [all_cls_logits[i] for i in (9, 6, 3)]
                    chosen_layer_logits = [self._classification_list[i](chosen_layer_cls_logits[i]).squeeze(-1) for i in range(3)]
                    chosen_layer_res_sel_losses = [self._criterion(chosen_layer_logits[i],
                                                                   batch_data['original']["res_sel"]["label"].float()) for i in range(3)]
                    chosen_layer_res_sel_losses = [x.masked_fill(mask, 0).mean() for x in chosen_layer_res_sel_losses]
                    res_sel_loss_list += chosen_layer_res_sel_losses
                    logits = torch.stack([logits] + chosen_layer_logits, dim=0).mean(dim=0)
                ####################### multi layer cls prediction ########################
            if original_contrastive:
                cls_logits_aug = get_bert_output('augment')[0]
                z = self._projection(cls_logits)
                z_aug = self._projection(cls_logits_aug)
                contrastive_loss_list += self._nt_xent_criterion(z, z_aug)

        if self.training and (new_response_selection or new_contrastive or supervised_contrastive):
            cls_logits_contras = get_bert_output('contras')[0]
            if new_response_selection:
                logits_contras = self._classification(cls_logits_contras)  # bs, 1
                logits_contras = logits_contras.squeeze(-1)
                res_sel_loss_contras = self._criterion(logits_contras, batch_data['contras']["res_sel"]["label"].float())
                mask = batch_data['contras']["res_sel"]["label"] == -1
                res_sel_loss_contras = res_sel_loss_contras.masked_fill(mask, 0).mean()
                res_sel_loss_list.append(res_sel_loss_contras)
                if less_positive_res_sel:  # less positive examples
                    cls_logits_sample = get_bert_output('sample')[0]
                    logits_sample = self._classification(cls_logits_sample)  # bs, 1
                    logits_sample = logits_sample.squeeze(-1)
                    res_sel_loss_sample = self._criterion(logits_sample, batch_data['sample']["res_sel"]["label"].float())
                    mask = batch_data['sample']["res_sel"]["label"] == -1
                    res_sel_loss_sample = res_sel_loss_sample.masked_fill(mask, 0).mean()
                    res_sel_loss_list.append(res_sel_loss_sample)
                    if three_class_classification:  # 3 class classification
                        logits_three_class_contras = torch.log_softmax(self._classification2(cls_logits_contras), dim=-1)
                        logits_three_class_sample = torch.log_softmax(self._classification2(cls_logits_sample), dim=1)
                        logits_three_class = torch.cat((logits_three_class_contras, logits_three_class_sample), dim=0)
                        label2 = torch.cat((batch_data['contras']["res_sel"]['label2'], batch_data['sample']["res_sel"]['label2']))
                        classification_loss_three_class = nn.functional.nll_loss(logits_three_class, label2, ignore_index=-1)
                        res_sel_loss_list.append(classification_loss_three_class)
            if new_contrastive:
                z_contras = self._projection(cls_logits_contras)
                cls_logits_contras_aug = get_bert_output('contras_aug')[0]
                z_contras_aug = self._projection(cls_logits_contras_aug)
                contrastive_loss_list += self._nt_xent_criterion(z_contras, z_contras_aug)
            if supervised_contrastive:
                sup_z_contras = self._projection_sup(get_bert_output('contras')[0])
                sup_z_contras_aug = self._projection_sup(get_bert_output('contras_aug')[0])
                sup_z_sample = self._projection_sup(get_bert_output('sample')[0])
                labels = torch.cat((batch_data['contras']["res_sel"]['label2'], batch_data['contras_aug']["res_sel"]['label2'], batch_data['sample']["res_sel"]['label2']))
                sup_z = torch.cat((sup_z_contras, sup_z_contras_aug, sup_z_sample), dim=0)
                new_sup_z = torch.cat([sup_z[labels == target_label] for target_label in (0, 1, 2)], dim=0)
                new_sup_z = torch.stack((new_sup_z[0::2], new_sup_z[1::2]), dim=1)
                new_labels = torch.cat([labels[labels == target_label] for target_label in (0, 1, 2)], dim=0)[0::2]
                sup_con_loss = self._sup_con_loss(new_sup_z, new_labels)
                contrastive_loss_list.append(sup_con_loss)

        ins_loss = del_loss = srch_loss = torch.Tensor([0]).to(device)
        if self.hparams.do_sent_insertion and self.training:
            ins_loss = self._bert_insertion(batch_data['extra']["ins"], batch_data['extra']["ins"]["label"]).mean()
        if self.hparams.do_sent_deletion and self.training:
            del_loss = self._bert_deletion(batch_data['extra']["del"], batch_data['extra']["del"]["label"]).mean()
        if self.hparams.do_sent_search and self.training:
            srch_loss = self._bert_search(batch_data['extra']["srch"], batch_data['extra']["srch"]["label"]).mean()
        res_sel_loss = torch.stack(res_sel_loss_list).mean()
        if len(contrastive_loss_list) == 0:
            contrastive_loss_list = [torch.Tensor([0]).to(res_sel_loss.device)]
        contrastive_loss = torch.stack(contrastive_loss_list).mean()
        return logits, (res_sel_loss, contrastive_loss, ins_loss, del_loss, srch_loss)
