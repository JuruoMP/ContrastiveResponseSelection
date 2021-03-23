import os
import copy

import torch
import torch.nn as nn
from models.bert.modeling_bert import BertModel, BertPreTrainedModel, BertEncoder, BertEmbeddings, BertPooler
from models.bert.modeling_bert import BertAttention, BertIntermediate, BertOutput


class BertClsMoe(nn.Module):
    def __init__(self, hparams):
        super(BertClsMoe, self).__init__()
        self.hparams = hparams

        pretrained_config = hparams.pretrained_config.from_pretrained(
            os.path.join(self.hparams.bert_pretrained_dir, self.hparams.bert_pretrained,
                         "%s-config.json" % self.hparams.bert_pretrained),
        )
        bert_model = BertMoeModel.from_pretrained(
            os.path.join(self.hparams.bert_pretrained_dir, self.hparams.bert_pretrained,
                         self.hparams.bert_checkpoint_path),
            config=pretrained_config, n_expert=hparams.n_expert
        )

        num_new_tok = 0
        if self.hparams.model_type.startswith("bert_base") or self.hparams.model_type.startswith("electra_base"):
            if self.hparams.do_eot:
                num_new_tok += 1
            # bert_post already has [EOT]

        bert_model.resize_token_embeddings(bert_model.config.vocab_size + num_new_tok)  # [EOT]
        self._model = bert_model

        self._classification = nn.Sequential(
            nn.Dropout(p=1 - self.hparams.dropout_keep_prob),
            nn.Linear(self.hparams.bert_hidden_dim, 1)
        )

        self._criterion = nn.BCEWithLogitsLoss()
        self.moe_criterion = MoeLoss()

    def forward(self, batch):
        logits, res_sel_loss, moe_loss = None, None, None

        if self.hparams.do_response_selection:
            outputs, all_expert_probs = self._model(
                batch['original']["res_sel"]["anno_sent"],
                token_type_ids=batch['original']["res_sel"]["segment_ids"],
                attention_mask=batch['original']["res_sel"]["attention_mask"]
            )
            bert_outputs = outputs[0]
            cls_logits = bert_outputs[:, 0, :]  # bs, bert_output_size
            logits = self._classification(cls_logits)  # bs, 1
            logits = logits.squeeze(-1)
            res_sel_loss = self._criterion(logits, batch['original']["res_sel"]["label"].float())

            expert_loss = sum(self.moe_criterion(x, batch['original']["res_sel"]["attention_mask"]) for x in all_expert_probs)

        return logits, (res_sel_loss, expert_loss, None, None)


class MoeLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, mask):
        batch_size, length, n_expert = x.size(0), x.size(1), x.size(2)
        # expert_attention_mask = mask.unsqueeze(-1).expand(x.size())

        xx = x[mask == 1]
        selection = xx.argmax(dim=-1)
        f = [torch.true_divide((selection == i).sum(), selection.size(0)) for i in range(n_expert)]
        p = [torch.true_divide(xx[selection == i, i].sum(), selection.size(0)) for i in range(n_expert)]
        loss = n_expert * sum(f[i] * p[i] for i in range(n_expert)) / xx.size(0)
        return loss


class BertMoeLayer(nn.Module):
    def __init__(self, config):
        assert hasattr(config, 'n_expert'), "Config param n_expert should be set in MoE mode."
        super().__init__()
        self.config = config
        self.attention = BertAttention(config)
        self.is_decoder = config.is_decoder
        if self.is_decoder:
            self.crossattention = BertAttention(config)
        self.router = nn.Linear(config.hidden_size, config.n_expert)
        self.intermediate = nn.ModuleList([BertIntermediate(config) for _ in range(config.n_expert)])
        self.output = BertOutput(config)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
    ):
        device = hidden_states.device
        batch_size, max_length = hidden_states.size(0), hidden_states.size(1)
        self_attention_outputs = self.attention(hidden_states, attention_mask, head_mask)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        if self.is_decoder and encoder_hidden_states is not None:
            cross_attention_outputs = self.crossattention(
                attention_output, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:]  # add cross attentions if we output attention weights

        routing_scores = torch.softmax(self.router(attention_output), dim=-1)
        routing_expert = routing_scores.argmax(dim=-1)
        flat_attention_output = attention_output.view(-1, attention_output.size(-1))
        flat_routing_expert = routing_expert.view(-1)
        flat_intermediate_output = torch.zeros(batch_size * max_length, self.config.intermediate_size).to(device)
        for i in range(self.config.n_expert):
            expert_intermediate_output = self.intermediate[i](flat_attention_output[flat_routing_expert == i])  # todo: warning: no expert working now
            flat_intermediate_output[flat_routing_expert == i] = expert_intermediate_output
        intermediate_output = flat_intermediate_output.view(batch_size, max_length, -1)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output, routing_scores,) + outputs
        return outputs


class BertMoeEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([BertMoeLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
    ):
        all_hidden_states = ()
        all_attentions = ()
        all_expert_probs = ()

        for i in range(self.config.num_hidden_layers):
            layer_module = self.layer[i]

            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states, attention_mask, head_mask[i], encoder_hidden_states, encoder_attention_mask
            )
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[2],)
            all_expert_probs = all_expert_probs + (layer_outputs[1], )

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs, all_expert_probs  # last-layer hidden state, (all hidden states), (all attentions)

    @classmethod
    def moe(cls, model):
        moe_encoder = BertMoeEncoder(model.config)
        for i in range(len(model.layer)):
            bert_layer = model.layer[i]
            moe_layer = moe_encoder.layer[i]
            moe_layer.attention.load_state_dict(bert_layer.attention.state_dict())
            if moe_layer.is_decoder:
                moe_layer.crossattention.load_state_dict(bert_layer.crossattention.state_dict())
            for j in range(moe_encoder.config.n_expert):
                moe_layer.intermediate[j].load_state_dict(bert_layer.intermediate.state_dict())
        return moe_encoder


class BertMoeModel(BertPreTrainedModel):
    def __init__(self, config, bert_model=None):
        assert hasattr(config, 'n_expert')
        super().__init__(config)
        self.config = config

        if bert_model is None:
            self.embeddings = BertEmbeddings(config)
            self.encoder = BertMoeEncoder(config)
            self.pooler = BertPooler(config)
            self.init_weights()
        else:
            self.embeddings = bert_model.embeddings
            self.encoder = BertMoeEncoder.moe(bert_model.encoder)
            self.pooler = bert_model.pooler

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
    ):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )
        encoder_outputs, all_expert_probs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        outputs = (sequence_output, pooled_output,) + encoder_outputs[
                                                      1:
                                                      ]  # add hidden_states and attentions if they are here
        return outputs, all_expert_probs  # sequence_output, pooled_output, (hidden_states), (attentions)

    @staticmethod
    def from_pretrained(*args, **kwargs):
        n_expert = kwargs['n_expert']
        del kwargs['n_expert']
        model = BertModel.from_pretrained(*args, **kwargs)
        model.config.n_expert = n_expert
        moe_model = BertMoeModel(model.config, model)
        return moe_model


if __name__ == '__main__':
    bert_moe_model = BertMoeModel.from_pretrained('bert-base-uncased', n_expert=3)
    a = 1
