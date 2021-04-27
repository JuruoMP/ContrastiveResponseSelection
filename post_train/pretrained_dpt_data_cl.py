import os

import torch
from torch import nn

from models.bert import modeling_bert, configuration_bert
from models.electra import modeling_electra, configuration_electra
from models.contrastive_loss import NTXentLoss, ConditionalNTXentLoss


class BertDPTDataCL(nn.Module):
    def __init__(self, hparams):
        super(BertDPTDataCL, self).__init__()
        self.hparams = hparams
        bert_config = configuration_bert.BertConfig.from_pretrained(
            os.path.join(self.hparams.bert_pretrained_dir, self.hparams.bert_pretrained,
                         "%s-config.json" % self.hparams.bert_pretrained),
        )
        bert_config.return_cls = True
        self._bert_model = modeling_bert.BertForPreTraining.from_pretrained(
            os.path.join(self.hparams.bert_pretrained_dir, self.hparams.bert_pretrained,
                         self.hparams.bert_checkpoint_path),
            config=bert_config
        )

        if self.hparams.do_eot:
            self._bert_model.resize_token_embeddings(self._bert_model.config.vocab_size + 1)  # [EOT]

        self.contrastive_criterion = ConditionalNTXentLoss(0.5, use_cosine_similarity=True)

    def forward(self, batch):
        bert_outputs1 = self._bert_model(
            input_ids=batch["input_ids"],
            token_type_ids=batch["token_type_ids"],
            attention_mask=batch["attention_mask"],
            masked_lm_labels=batch["masked_lm_labels"],
            next_sentence_label=batch["next_sentence_labels"],
        )
        bert_outputs2 = self._bert_model(
            input_ids=batch["input_ids"],
            token_type_ids=batch["token_type_ids"],
            attention_mask=batch["attention_mask"],
            masked_lm_labels=batch["masked_lm_labels"],
            next_sentence_label=batch["next_sentence_labels"],
        )
        mlm_loss1, nsp_loss1, prediction_scores1, seq_relationship_score1, cls_vec1 = bert_outputs1[:5]
        mlm_loss2, nsp_loss2, prediction_scores2, seq_relationship_score2, cls_vec2 = bert_outputs2[:5]

        contras_bert_outputs1 = self._bert_model.bert(
            input_ids=batch["contras_input_ids"],
            token_type_ids=batch["contras_token_type_ids"],
            attention_mask=batch["contras_attention_mask"]
        )
        contras_bert_outputs2 = self._bert_model.bert(
            input_ids=batch["contras_input_ids"],
            token_type_ids=batch["contras_token_type_ids"],
            attention_mask=batch["contras_attention_mask"]
        )
        contras_cls_vec1 = contras_bert_outputs1[1]
        contras_cls_vec2 = contras_bert_outputs2[1]

        mlm_loss = (mlm_loss1 + mlm_loss2) / 2
        nsp_loss = (nsp_loss1 + nsp_loss2) / 2

        cl_loss = self.contrastive_criterion(torch.cat((cls_vec1, contras_cls_vec1), dim=0),
                                             torch.cat((cls_vec2, contras_cls_vec2), dim=0))

        return None, mlm_loss, nsp_loss, cl_loss


class ElectraDPTDataCL(nn.Module):
    def __init__(self, hparams):
        super(ElectraDPTDataCL, self).__init__()
        self.hparams = hparams

        electra_gen_config = configuration_electra.ElectraConfig.from_pretrained(
            os.path.join(self.hparams.bert_pretrained_dir, self.hparams.bert_pretrained,
                         "%s-config.json" % self.hparams.electra_gen_config),
        )
        self._electra_gen = modeling_electra.ElectraForMaskedLM.from_pretrained(
            os.path.join(self.hparams.bert_pretrained_dir, self.hparams.bert_pretrained,
                         self.hparams.electra_gen_ckpt_path),
            config=electra_gen_config
        )

        electra_disc_config = configuration_electra.ElectraConfig.from_pretrained(
            os.path.join(self.hparams.bert_pretrained_dir, self.hparams.bert_pretrained,
                         "%s-config.json" % self.hparams.bert_pretrained),
        )
        self._electra_disc = modeling_electra.ElectraForPreTraining.from_pretrained(
            os.path.join(self.hparams.bert_pretrained_dir, self.hparams.bert_pretrained,
                         self.hparams.bert_checkpoint_path),
            config=electra_disc_config
        )

        print(
            self._electra_gen.electra.embeddings.word_embeddings.weight == self._electra_disc.electra.embeddings.word_embeddings.weight)

        # tie weights (discriminator and generator)
        if self.hparams.do_eot:
            print("Adding Special Token [EOT]")
            self._electra_gen.resize_token_embeddings(self._electra_gen.config.vocab_size + 1)  # [EOT]
            print("gen embeddings size : ", self._electra_gen.electra.embeddings)
            print("gen output embeddings size : ", self._electra_gen.get_output_embeddings())

            print("###generator word embedding comparison between input and output###")
            print(
                self._electra_gen.electra.embeddings.word_embeddings.weight == self._electra_gen.get_output_embeddings().weight)
            print("###word embedding comparison between discriminator and generator###")
            print(
                self._electra_gen.electra.embeddings.word_embeddings.weight == self._electra_gen.generator_lm_head.weight)

            self._electra_disc.electra.embeddings = self._electra_gen.electra.embeddings
            print("disc embeddings size : ", self._electra_disc.electra.embeddings)

    def forward(self, batch):
        # print(torch.sum(self._electra_gen.electra.embeddings.word_embeddings.weight == self._electra_disc.electra.embeddings.word_embeddings.weight))
        # print(torch.sum(self._electra_gen.electra.embeddings.position_embeddings.weight == self._electra_disc.electra.embeddings.position_embeddings.weight))
        # print(torch.sum(self._electra_gen.electra.embeddings.token_type_embeddings.weight == self._electra_disc.electra.embeddings.token_type_embeddings.weight))
        # print(torch.sum(self._electra_gen.electra.embeddings.LayerNorm.weight == self._electra_disc.electra.embeddings.LayerNorm.weight))
        # print(torch.sum(self._electra_gen.electra.embeddings.LayerNorm.bias == self._electra_disc.electra.embeddings.LayerNorm.bias))

        # generator : input_ids, attention_mask, token_type_ids, masked_lm_labels
        genearator_outputs = self._electra_gen(
            input_ids=batch["masked_input_ids"],
            token_type_ids=batch["token_type_ids"],
            attention_mask=batch["attention_mask"],
            masked_lm_labels=batch["masked_lm_labels"],
        )
        mlm_loss, prediction_scores = genearator_outputs[:2]
        gen_loss = mlm_loss

        gen_predictons = torch.max(prediction_scores, dim=-1)[1]
        # compare to the original input_ids -> original 0, replaced 1
        disc_labels = (gen_predictons != batch["input_ids"]).int()

        discriminator_outputs = self._electra_disc(
            input_ids=gen_predictons,
            token_type_ids=batch["token_type_ids"],
            attention_mask=batch["attention_mask"],
            labels=disc_labels,
        )
        disc_loss = discriminator_outputs[0]

        # disc_loss => 50
        electra_loss = self.hparams.electra_disc_ratio * disc_loss + gen_loss

        return electra_loss, mlm_loss, None


class ElectraNSPDPTDataCL(nn.Module):
    def __init__(self, hparams):
        super(ElectraNSPDPTDataCL, self).__init__()
        self.hparams = hparams

        electra_gen_config = configuration_electra.ElectraConfig.from_pretrained(
            os.path.join(self.hparams.bert_pretrained_dir, self.hparams.bert_pretrained,
                         "%s-config.json" % self.hparams.electra_gen_config),
        )
        self._electra_gen = modeling_electra.ElectraForMaskedLM.from_pretrained(
            os.path.join(self.hparams.bert_pretrained_dir, self.hparams.bert_pretrained,
                         self.hparams.electra_gen_ckpt_path),
            config=electra_gen_config
        )

        electra_disc_config = configuration_electra.ElectraConfig.from_pretrained(
            os.path.join(self.hparams.bert_pretrained_dir, self.hparams.bert_pretrained,
                         "%s-config.json" % self.hparams.bert_pretrained),
        )
        self._electra_disc = modeling_electra.ElectraForPreTrainingNSP.from_pretrained(
            os.path.join(self.hparams.bert_pretrained_dir, self.hparams.bert_pretrained,
                         self.hparams.bert_checkpoint_path),
            config=electra_disc_config
        )

        print(
            self._electra_gen.electra.embeddings.word_embeddings.weight == self._electra_disc.electra.embeddings.word_embeddings.weight)

        # tie weights (discriminator and generator)
        if self.hparams.do_eot:
            print("Adding Special Token [EOT]")
            self._electra_gen.resize_token_embeddings(self._electra_gen.config.vocab_size + 1)  # [EOT]
            print("gen embeddings size : ", self._electra_gen.electra.embeddings)
            print("gen output embeddings size : ", self._electra_gen.get_output_embeddings())

            print("###generator word embedding comparison between input and output###")
            print(
                self._electra_gen.electra.embeddings.word_embeddings.weight == self._electra_gen.get_output_embeddings().weight)
            print("###word embedding comparison between discriminator and generator###")
            print(
                self._electra_gen.electra.embeddings.word_embeddings.weight == self._electra_gen.generator_lm_head.weight)

            self._electra_disc.electra.embeddings = self._electra_gen.electra.embeddings
            print("disc embeddings size : ", self._electra_disc.electra.embeddings)

    def forward(self, batch):
        # print(torch.sum(self._electra_gen.electra.embeddings.word_embeddings.weight == self._electra_disc.electra.embeddings.word_embeddings.weight))
        # print(torch.sum(self._electra_gen.electra.embeddings.position_embeddings.weight == self._electra_disc.electra.embeddings.position_embeddings.weight))
        # print(torch.sum(self._electra_gen.electra.embeddings.token_type_embeddings.weight == self._electra_disc.electra.embeddings.token_type_embeddings.weight))
        # print(torch.sum(self._electra_gen.electra.embeddings.LayerNorm.weight == self._electra_disc.electra.embeddings.LayerNorm.weight))
        # print(torch.sum(self._electra_gen.electra.embeddings.LayerNorm.bias == self._electra_disc.electra.embeddings.LayerNorm.bias))

        # generator : input_ids, attention_mask, token_type_ids, masked_lm_labels
        genearator_outputs = self._electra_gen(
            input_ids=batch["input_ids"],
            token_type_ids=batch["token_type_ids"],
            attention_mask=batch["attention_mask"],
            masked_lm_labels=batch["masked_lm_labels"]
        )

        mlm_loss, prediction_scores = genearator_outputs[:2]
        gen_loss = mlm_loss

        gen_predictons = torch.max(prediction_scores, dim=-1)[1]
        # compare to the original input_ids -> original 0, replaced 1
        disc_labels = (gen_predictons != batch["input_ids"]).int()

        discriminator_outputs = self._electra_disc(
            input_ids=gen_predictons,
            token_type_ids=batch["token_type_ids"],
            attention_mask=batch["attention_mask"],
            labels=disc_labels,
            next_sentence_label=batch["next_sentence_labels"]
        )
        disc_loss, nsp_loss = discriminator_outputs[:2]

        # disc_loss => 50
        electra_loss = self.hparams.electra_disc_ratio * disc_loss + gen_loss

        return electra_loss, mlm_loss, nsp_loss
