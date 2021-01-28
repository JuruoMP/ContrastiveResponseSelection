import os
import torch
import torch.nn as nn
from torch.cuda import amp

from models.bert_insertion import BertInsertion
from models.bert_deletion import BertDeletion
from models.bert_search import BertSearch
from models.contrastive_loss import NTXentLoss, ConditionalNTXentLoss, DynamicNTXentLoss


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

        self._projection = nn.Sequential(
            nn.Linear(self.hparams.bert_hidden_dim, self.hparams.projection_dim),
            nn.ReLU(),
            nn.Linear(self.hparams.projection_dim, self.hparams.projection_dim)
        )

        self._criterion = nn.BCEWithLogitsLoss()
        if self.hparams.use_batch_negative:
            self._nt_xent_criterion = NTXentLoss(temperature=0.5, use_cosine_similarity=True)
        self._nt_xent_criterion = DynamicNTXentLoss(temperature=0.5, use_cosine_similarity=True)
        self.hinge_lambda = 0.4
        self.return_augment = False

    @amp.autocast()
    def forward(self, batch_data):
        contrastive_loss, hinge_loss = None, None

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
        res_sel_loss = self._criterion(logits, batch["res_sel"]["label"])

        if self.hparams.do_contrastive and self.training:
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

            batch_soft_logits = None
            if self.hparams.use_soft_logits:
                soft_logits, soft_logits_aug = batch['res_sel']['soft_logits'], batch_aug['res_sel']['soft_logits']  # n_example, n_example
                if len(soft_logits.size()) == 1:
                    batch_soft_logits = torch.stack((soft_logits, soft_logits_aug), dim=1).view(-1, 4)  # n_query * 4
                elif len(soft_logits.size()) == 2:
                    soft_logits = soft_logits.split(2, dim=0)
                    soft_logits_aug = soft_logits_aug.split(2, dim=0)
                    batch_soft_logits = [torch.cat((x, y), dim=0) for x, y in zip(soft_logits, soft_logits_aug)]
                else:
                    raise Exception('Invalid soft logits')
            contrastive_loss, hinge_loss = self._nt_xent_criterion(z, z_aug, batch_soft_logits=batch_soft_logits)

        if not self.return_augment:  # todo: useless
            return logits, (res_sel_loss, contrastive_loss, hinge_loss)
        else:
            return (logits, logits_aug), (res_sel_loss, contrastive_loss, hinge_loss)
