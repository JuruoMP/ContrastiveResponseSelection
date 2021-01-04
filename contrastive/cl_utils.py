import os
import multiprocessing

import torch
from tqdm import tqdm


class ContrastiveUtils:
    def __init__(self, lang='en'):
        if lang == 'en':
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
        elif lang == 'zh':
            from contrastive.eda.augment_util import EDAUtil
            self.eda_util = EDAUtil(num_augment=2, alpha=0.05, lang='zh')

    def eda_augment(self, x):
        return self.eda_util.augment_sentence(x)

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


class ResponseRetriever:
    def __init__(self):
        super().__init__()

    @staticmethod
    def load_file(file_path):
        dialogs = []
        with open(file_path, 'r', encoding='utf-8') as fr:
            while True:
                line = fr.readline()
                if not line:
                    break
                label, turns = line.strip().split('\t', 1)
                if label == '1':
                    turns = turns.split('\t')
                    dialogs.append((turns[:-1], turns[-1]))
                # if len(dialogs) >= 100:
                #     break
        return dialogs

    @staticmethod
    def retrieve_response(context, responses, context_idx=-1):
        best_res_idx, best_score = None, -1
        for response_idx, res in enumerate(responses):
            if response_idx == context_idx:
                continue
            score = ResponseRetriever.jaccard_coefficient(context, res)
            if score > best_score:
                best_score = score
                best_res_idx = response_idx
        return best_res_idx

    @staticmethod
    def retrieve_batch_response(batch_context, responses, batch_context_idx, output_file):
        print(f'Start creating {output_file}')
        batch_response = []
        iterator = zip(batch_context_idx, batch_context)
        if '_0.txt' in output_file:
            iterator = tqdm(iterator, total=len(batch_context))
        for context_idx, context in iterator:
            response = ResponseRetriever.retrieve_response(context, responses, context_idx)
            batch_response.append(response)
        with open(output_file, 'w', encoding='utf-8') as fw:
            for response in batch_response:
                fw.write(str(response) + '\n')
        print(f'File {output_file} created successfully')

    @staticmethod
    def retrieve_dataset_response(contexts, responses, n_process=500):
        original_responses = responses
        contexts = [set(' '.join(x).split()) for x in contexts]  # use context to retrieve
        # contexts = [set(x.split()) for x in responses]  # use response to retrieve
        responses = [set(x.split()) for x in responses]
        pool = multiprocessing.Pool(processes=24)
        split_size = len(contexts) // n_process
        for i in range(n_process):
            range_st, range_ed = i * split_size, (i + 1) * split_size
            batch_context = contexts[range_st:range_ed]
            batch_context_idx = list(range(range_st, range_ed))
            output_file_path = f'cache/retrieve_{i}.txt'
            pool.apply_async(ResponseRetriever.retrieve_batch_response, (batch_context, responses, batch_context_idx, output_file_path,))
        pool.close()
        pool.join()

        print('Generating final retrieved response file')
        all_response_lines = []
        for i in range(n_process):
            output_file_path = f'cache/retrieve_{i}.txt'
            all_response_lines += [int(x.strip()) for x in open(output_file_path, 'r', encoding='utf-8').readlines()]
        with open('data/ubuntu_corpus_v1/retrieve_response.txt', 'w', encoding='utf-8') as fw:
            for response_idx in tqdm(all_response_lines):
                fw.write(original_responses[response_idx] + '\n')

    @staticmethod
    def jaccard_coefficient(tokens1, tokens2):
        return len(tokens1 & tokens2) / len(tokens1 | tokens2)


if __name__ == '__main__':
    # cl_util = ContrastiveUtils()
    # uttrs = ['who am i', 'do you know which one is better', 'please help me to acceluate this program']
    # ret = cl_util.batch_back_translation_augmentation(uttrs)
    # print(ret)
    # cl_util = ContrastiveUtils(lang='zh')
    # ret = cl_util.eda_augment('我 去 不 早 说 发韵 达 能 到 我家 那儿 我 就 能 拿到'.replace(' ', ''))
    # print(ret)
    ubuntu_data_path = 'data/ubuntu_corpus_v1/train.txt'
    response_retriever = ResponseRetriever()
    # response_retriever.retrieve_batch_response(response_retriever.contexts, [i for i in range(500000)], 'tmp/0.txt')
    dialogs = response_retriever.load_file(ubuntu_data_path)
    contexts = [x[0] for x in dialogs]
    responses = [x[1] for x in dialogs]
    response_retriever.retrieve_dataset_response(contexts, responses)
