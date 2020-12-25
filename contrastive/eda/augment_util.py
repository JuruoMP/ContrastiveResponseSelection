from .en_eda.code.eda import eda as en_eda
from .zh_eda.code.eda import eda as zh_eda


class EDAUtil:
    def __init__(self, num_augment, alpha, lang='zh'):
        self.num_augment = num_augment
        self.alpha = alpha
        self.lang = lang
        if lang == 'zh':
            self.eda = zh_eda
        elif lang == 'en':
            self.eda = en_eda

    def augment_sentences(self, sentences):
        return [self.augment_sentence(sentence) for sentence in sentences]

    def augment_sentence(self, sentence):
        return self.eda(sentence, alpha_sr=self.alpha, alpha_ri=self.alpha,
                        alpha_rs=self.alpha, p_rd=self.alpha, num_aug=self.num_augment)
