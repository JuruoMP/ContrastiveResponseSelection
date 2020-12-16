import os
import torch.nn as nn

from models.bert_insertion import BertInsertion
from models.bert_deletion import BertDeletion
from models.bert_search import BertSearch
from models.contrastive_loss import NTXentLoss


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

        num_new_tok = 0
        if self.hparams.model_type.startswith("bert_base") or self.hparams.model_type.startswith("electra_base"):
            if self.hparams.do_eot:
                num_new_tok += 1
            # bert_post already has [EOT]

        if self.hparams.do_sent_insertion:
            num_new_tok += 1  # [INS]
        if self.hparams.do_sent_deletion:
            num_new_tok += 1  # [INS]
        if self.hparams.do_sent_search:
            num_new_tok += 1  # [SRCH]

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

        if self.hparams.do_sent_insertion:
            self._bert_insertion = BertInsertion(hparams, self._model)
        if self.hparams.do_sent_deletion:
            self._bert_deletion = BertDeletion(hparams, self._model)
        if self.hparams.do_sent_search:
            self._bert_search = BertSearch(hparams, self._model)

        self._criterion = nn.BCEWithLogitsLoss()
        self._nt_xent_criterion = NTXentLoss(temperature=0.5, use_cosine_similarity=True)

    def forward(self, batch):
        logits, res_sel_loss, ins_loss, del_loss, srch_loss, contrastive_loss = None, None, None, None, None, None

        if self.hparams.do_response_selection:
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

        if self.hparams.do_sent_insertion and (self.training or self.hparams.pca_visualization):
            ins_loss = self._bert_insertion(batch["ins"], batch["res_sel"]["label"])
        if self.hparams.do_sent_deletion and (self.training or self.hparams.pca_visualization):
            del_loss = self._bert_deletion(batch["del"], batch["res_sel"]["label"])
        if self.hparams.do_sent_search and (self.training or self.hparams.pca_visualization):
            srch_loss = self._bert_search(batch["srch"], batch["res_sel"]["label"])

        if self.hparams.do_contrastive and self.training:
            outputs_aug1 = self._model(
                batch["res_sel_aug1"]["anno_sent"],
                token_type_ids=batch["res_sel_aug1"]["segment_ids"],
                attention_mask=batch["res_sel_aug1"]["attention_mask"]
            )
            outputs_aug2 = self._model(
                batch["res_sel_aug2"]["anno_sent"],
                token_type_ids=batch["res_sel_aug2"]["segment_ids"],
                attention_mask=batch["res_sel_aug2"]["attention_mask"]
            )
            bert_outputs_aug1 = outputs_aug1[0]
            bert_outputs_aug2 = outputs_aug2[0]
            cls_aug1 = bert_outputs_aug1[:, 0, :]
            cls_aug2 = bert_outputs_aug2[:, 0, :]
            z_aug1 = self._projection(cls_aug1)
            z_aug2 = self._projection(cls_aug2)
            contrastive_loss = self._nt_xent_criterion(z_aug1, z_aug2)

        return logits, (res_sel_loss, ins_loss, del_loss, srch_loss, contrastive_loss)
