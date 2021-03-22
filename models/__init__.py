from models.bert_cls_cl_v6 import BertCls
from models.bert_eot import BertEOT
from models.bert_insertion import BertInsertion
# from models.bert_cls_moe import BertClsMoe


def Model(hparams, *args, **kwargs):
    name_model_map = {
        "bert_base": BertCls,
        "bert_post": BertCls,
        # "bert_post_moe": BertClsMoe,

        "electra_base": BertCls,
        "electra_post": BertCls,

        "bert_base_eot": BertEOT,
        "bert_post_eot": BertEOT,
    }

    return name_model_map[hparams.model_type](hparams, *args, **kwargs)
