import os
import random
import torch
import torch.nn as nn
from torch.cuda import amp

from models.contrastive_loss import NTXentLoss, DynamicNTXentLoss, SupConLoss
import global_variables
from models.bert_insertion import BertInsertion


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
            if self.hparams.do_sent_insertion:
                num_new_tok += 1  # [INS]
                self._bert_insertion = BertInsertion(hparams, self._model)

        self._model.resize_token_embeddings(self._model.config.vocab_size + num_new_tok)  # [EOT]

        self._classification = nn.Sequential(
            nn.Dropout(p=1 - self.hparams.dropout_keep_prob),
            nn.Linear(self.hparams.bert_hidden_dim, 1)
        )

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
        def get_bert_output(name, return_hidden=False):
            if bert_output_cache.get(name) is not None:
                return bert_output_cache.get(name)
            else:
                batch = batch_data[name]
                bert_outputs, _, hidden_states = self._model(
                    batch["res_sel"]["anno_sent"],
                    token_type_ids=batch["res_sel"]["segment_ids"],
                    attention_mask=batch["res_sel"]["attention_mask"]
                )
                cls_logits = bert_outputs[:, 0, :]  # bs, bert_output_size
                bert_output_cache[name] = cls_logits
                if not return_hidden:
                    return cls_logits
                else:
                    return cls_logits, hidden_states

        device = batch_data['original']['res_sel']['anno_sent'].device
        if self.training:  # training
            original_response_selection, original_contrastive = True, True
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
            cls_logits = get_bert_output('original')
            if original_response_selection:
                logits = self._classification(cls_logits)  # bs, 1
                logits = logits.squeeze(-1)
                res_sel_losses = self._criterion(logits, batch_data['original']["res_sel"]["label"].float())
                mask = batch_data['original']["res_sel"]["label"] == -1
                res_sel_loss = res_sel_losses.masked_fill(mask, 0).mean()
                res_sel_loss_list.append(res_sel_loss)
            if original_contrastive:
                cls_logits_aug = get_bert_output('augment')
                z = self._projection(cls_logits)
                z_aug = self._projection(cls_logits_aug)
                contrastive_loss_list += self._nt_xent_criterion(z, z_aug)

        if self.training and (new_response_selection or new_contrastive or supervised_contrastive):
            cls_logits_contras = get_bert_output('contras')
            if new_response_selection:
                logits_contras = self._classification(cls_logits_contras)  # bs, 1
                logits_contras = logits_contras.squeeze(-1)
                res_sel_loss_contras = self._criterion(logits_contras, batch_data['contras']["res_sel"]["label"].float())
                mask = batch_data['contras']["res_sel"]["label"] == -1
                res_sel_loss_contras = res_sel_loss_contras.masked_fill(mask, 0).mean()
                res_sel_loss_list.append(res_sel_loss_contras)
                if less_positive_res_sel:  # less positive examples
                    cls_logits_sample = get_bert_output('sample')
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
                cls_logits_contras_aug = get_bert_output('contras_aug')
                z_contras_aug = self._projection(cls_logits_contras_aug)
                contrastive_loss_list += self._nt_xent_criterion(z_contras, z_contras_aug)
            if supervised_contrastive:
                sup_z_contras = self._projection_sup(get_bert_output('contras'))
                sup_z_contras_aug = self._projection_sup(get_bert_output('contras_aug'))
                sup_z_sample = self._projection_sup(get_bert_output('sample'))
                labels = torch.cat((batch_data['contras']["res_sel"]['label2'], batch_data['contras_aug']["res_sel"]['label2'], batch_data['sample']["res_sel"]['label2']))
                sup_z = torch.cat((sup_z_contras, sup_z_contras_aug, sup_z_sample), dim=0)
                new_sup_z = torch.cat([sup_z[labels == target_label] for target_label in (0, 1, 2)], dim=0)
                new_sup_z = torch.stack((new_sup_z[0::2], new_sup_z[1::2]), dim=1)
                new_labels = torch.cat([labels[labels == target_label] for target_label in (0, 1, 2)], dim=0)[0::2]
                sup_con_loss = self._sup_con_loss(new_sup_z, new_labels)
                contrastive_loss_list.append(sup_con_loss)
        if self.hparams.do_sent_insertion and self.training:
            ins_loss = self._bert_insertion(batch_data['extra']["ins"], batch_data['extra']["ins"]["label"])
        else:
            ins_loss = torch.Tensor([0]).to(device)
        res_sel_loss = torch.stack(res_sel_loss_list).mean()
        if len(contrastive_loss_list) == 0:
            contrastive_loss_list = [torch.Tensor([0]).to(res_sel_loss.device)]
        contrastive_loss = torch.stack(contrastive_loss_list).mean()
        return logits, (res_sel_loss, contrastive_loss, ins_loss)
