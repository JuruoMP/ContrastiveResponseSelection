import os
import random
import torch
import torch.nn as nn
from torch.cuda import amp

from models.contrastive_loss import NTXentLoss, DynamicNTXentLoss
import global_variables


class BertCls(nn.Module):
    def __init__(self, hparams):
        super(BertCls, self).__init__()
        self.hparams = hparams

        pretrained_config = hparams.pretrained_config.from_pretrained(
            os.path.join(self.hparams.bert_pretrained_dir, self.hparams.bert_pretrained,
                         "%s-config.json" % self.hparams.bert_pretrained),
        )
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

        self._criterion = nn.BCEWithLogitsLoss(reduction='none')
        if self.hparams.use_batch_negative:
            self._nt_xent_criterion = NTXentLoss(temperature=0.5, use_cosine_similarity=True)
        self._nt_xent_criterion = DynamicNTXentLoss(temperature=0.5, use_cosine_similarity=True)

    @amp.autocast()
    def forward(self, batch_data):
        batch = batch_data['original']
        outputs = self._model(
            batch["res_sel"]["anno_sent"],
            token_type_ids=batch["res_sel"]["segment_ids"],
            attention_mask=batch["res_sel"]["attention_mask"]
        )
        bert_outputs = outputs[0]
        cls_logits = bert_outputs[:, 0, :]  # bs, bert_output_size
        logits = self._classification(cls_logits)  # bs, 1
        logits = logits.squeeze(-1)
        if self.hparams.do_response_selection:
            res_sel_losses = self._criterion(logits, batch["res_sel"]["label"].float())
            mask = batch["res_sel"]["label"] == -1
            res_sel_loss = res_sel_losses.masked_fill(mask, 0).mean()
            # 3 class classification
            logits_three_class = torch.log_softmax(self._classification2(cls_logits), dim=-1)
            if 'label2' in batch["res_sel"]:
                classification_loss = nn.functional.nll_loss(logits_three_class, batch["res_sel"]['label2'])
                res_sel_loss = res_sel_loss + classification_loss

        contrastive_loss = []
        if self.hparams.do_contrastive and self.training:
            do_origin_contras = False #True if random.random() > 0.1 * global_variables.epoch else False
            do_extra_contras = not do_origin_contras
            if do_origin_contras:
                batch_aug = batch_data['augment']
                outputs_aug = self._model(
                    batch_aug["res_sel"]["anno_sent"],
                    token_type_ids=batch_aug["res_sel"]["segment_ids"],
                    attention_mask=batch_aug["res_sel"]["attention_mask"]
                )
                bert_outputs_aug = outputs_aug[0]
                cls_logits_aug = bert_outputs_aug[:, 0, :]
                z = self._projection(cls_logits)
                z_aug = self._projection(cls_logits_aug)

                logits_aug = self._classification(cls_logits_aug)  # bs, 1
                logits_aug = logits_aug.squeeze(-1)
                if self.hparams.do_augment_response_selection:
                    res_sel_loss = (res_sel_loss + self._criterion(logits_aug, batch_aug["res_sel"]["label"])) / 2
                contrastive_loss += self._nt_xent_criterion(z, z_aug)
            if do_extra_contras:
                batch_contras = batch_data['contras']
                outputs_contras = self._model(
                    batch_contras["res_sel"]["anno_sent"],
                    token_type_ids=batch_contras["res_sel"]["segment_ids"],
                    attention_mask=batch_contras["res_sel"]["attention_mask"]
                )
                bert_outputs_contras = outputs_contras[0]
                cls_logits_contras = bert_outputs_contras[:, 0, :]
                z_contras = self._projection(cls_logits_contras)
                batch_contras_aug = batch_data['contras_aug']
                outputs_contras_aug = self._model(
                    batch_contras_aug["res_sel"]["anno_sent"],
                    token_type_ids=batch_contras_aug["res_sel"]["segment_ids"],
                    attention_mask=batch_contras_aug["res_sel"]["attention_mask"]
                )
                bert_outputs_contras_aug = outputs_contras_aug[0]
                cls_logits_contras_aug = bert_outputs_contras_aug[:, 0, :]
                z_contras_aug = self._projection(cls_logits_contras_aug)
                contrastive_loss += self._nt_xent_criterion(z_contras, z_contras_aug)
                if True:  # train contras example with response selection loss
                    # res_sel loss from contras examples
                    logits_contras = self._classification(cls_logits_contras)  # bs, 1
                    logits_contras = logits_contras.squeeze(-1)
                    res_sel_losses_contras = self._criterion(logits_contras, batch_contras["res_sel"]["label"].float())
                    mask = batch_contras["res_sel"]["label"] == -1
                    res_sel_loss_contras = res_sel_losses_contras.masked_fill(mask, 0).mean()

                    # res_sel loss from sample examples
                    batch_sample = batch_data['sample']
                    outputs_sample = self._model(
                        batch_sample["res_sel"]["anno_sent"],
                        token_type_ids=batch_sample["res_sel"]["segment_ids"],
                        attention_mask=batch_sample["res_sel"]["attention_mask"]
                    )
                    bert_outputs_sample = outputs_sample[0]
                    cls_logits_sample = bert_outputs_sample[:, 0, :]  # bs, bert_output_size
                    logits_sample = self._classification(cls_logits_sample)  # bs, 1
                    logits_sample = logits_sample.squeeze(-1)
                    res_sel_losses_sample = self._criterion(logits_sample, batch_sample["res_sel"]["label"].float())
                    mask = batch_sample["res_sel"]["label"] == -1
                    res_sel_losses_sample = res_sel_losses_sample.masked_fill(mask, 0).mean()
                    res_sel_loss = torch.stack((res_sel_loss, res_sel_loss_contras, res_sel_losses_sample)).mean()

        return logits, (res_sel_loss, contrastive_loss)
