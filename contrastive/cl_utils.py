from tqdm import tqdm
import torch


class ContrastiveUtils:
    def __init__(self):
        en2de = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-de.single_model', tokenizer='moses',
                               bpe='fastbpe').cuda()
        de2en = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.de-en.single_model', tokenizer='moses',
                               bpe='fastbpe').cuda()
        en2ru = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-ru.single_model', tokenizer='moses',
                               bpe='fastbpe').cuda()
        ru2en = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.ru-en.single_model', tokenizer='moses',
                               bpe='fastbpe').cuda()

        self.translation_model = {
            'de': (en2de, de2en),
            'ru': (en2ru, ru2en)
        }

    def back_translation_augmentation(self, x):
        translations = []
        for language in self.translation_model:
            model1, model2 = self.translation_model.get(language.lower())
            y = model1.translate(x)
            x_prime = model2.translate(y)
            translations.append(x_prime)
        return translations

    def batch_back_translation_augmentation(self, x_list, batch_size=200):
        translations = []
        for language in self.translation_model:
            model1, model2 = self.translation_model.get(language.lower())
            x_prime_list = []
            for i in tqdm(range(0, len(x_list), batch_size)):
                xx_list = x_list[i: i + batch_size]
                yy_list = model1.translate(xx_list)
                xx_prime_list = model2.translate(yy_list)
                x_prime_list.extend(xx_prime_list)
            translations.append(x_prime_list)
        return list(zip(*translations))


if __name__ == '__main__':
    cl_util = ContrastiveUtils()
    uttrs = ['who am i', 'do you know which one is better', 'please help me to acceluate this program']
    ret = cl_util.batch_back_translation_augmentation(uttrs)
    print(ret)