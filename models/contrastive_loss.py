import numpy as np
import torch
import torch.nn as nn


class NTXentLoss(nn.Module):
    def __init__(self, temperature, use_cosine_similarity):
        super().__init__()
        self.temperature = temperature
        self.softmax = torch.nn.Softmax(dim=-1)
        self.similarity_function = torch.nn.CosineSimilarity(dim=-1) if use_cosine_similarity else self._dot_similarity
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_correlated_mask(self, batch_size):
        diag = np.eye(2 * batch_size)
        l1 = np.eye((2 * batch_size), 2 * batch_size, k=-batch_size)
        l2 = np.eye((2 * batch_size), 2 * batch_size, k=batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).bool()
        return mask

    @staticmethod
    def _dot_similarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def forward(self, zis, zjs):
        device = zis.device
        batch_size = zis.size(0)
        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(representations.unsqueeze(1), representations.unsqueeze(0))

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, batch_size)
        r_pos = torch.diag(similarity_matrix, -batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * batch_size, 1)

        mask_samples_from_same_repr = self._get_correlated_mask(batch_size).to(device)
        negatives = similarity_matrix[mask_samples_from_same_repr].view(2 * batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * batch_size).to(device).long()
        loss = self.criterion(logits, labels)

        return loss / (2 * batch_size)


class ConditionalNTXentLoss(nn.Module):
    def __init__(self, temperature, use_cosine_similarity):
        super().__init__()
        self.temperature = temperature
        self.softmax = torch.nn.Softmax(dim=-1)
        self.similarity_function = torch.nn.CosineSimilarity(dim=-1) if use_cosine_similarity else self._dot_similarity
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_correlated_mask(self, batch_size):
        diag = np.eye(2 * batch_size)
        l1 = np.eye((2 * batch_size), 2 * batch_size, k=-batch_size)
        l2 = np.eye((2 * batch_size), 2 * batch_size, k=batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).bool()
        return mask

    @staticmethod
    def _dot_similarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        return v

    def forward(self, zis, zjs):
        device = zis.device
        batch_size = zis.size(0)
        zis_list = zis.split(2, dim=0)
        zjs_list = zjs.split(2, dim=0)

        loss = 0
        for zis, zjs in zip(zis_list, zjs_list):
            representations = torch.cat([zjs, zis], dim=0)

            similarity_matrix = self.similarity_function(representations.unsqueeze(1), representations.unsqueeze(0))

            # filter out the scores from the positive samples
            l_pos = torch.diag(similarity_matrix, 2)
            r_pos = torch.diag(similarity_matrix, -2)
            positives = torch.cat([l_pos, r_pos]).view(2 * 2, 1)

            mask_samples_from_same_repr = self._get_correlated_mask(2).to(device)
            negatives = similarity_matrix[mask_samples_from_same_repr].view(2 * 2, -1)

            logits = torch.cat((positives, negatives), dim=1)
            logits /= self.temperature

            labels = torch.zeros(2 * 2).to(device).long()
            loss += self.criterion(logits, labels)

        return loss / (2 * batch_size)


class DynamicNTXentLoss(ConditionalNTXentLoss):
    def criterion(self, logits, soft_labels, reduction='sum'):
        # standard_ce_loss = torch.nn.functional.cross_entropy(logits, soft_labels, reduction='sum')
        logits = torch.exp(logits)
        ps = logits / logits.sum(dim=1, keepdim=True).expand(logits.size())
        loss = -(torch.log(ps) * soft_labels)
        if reduction == 'sum':
            loss = loss.sum()
        else:
            loss = loss.mean()
        return loss

    def forward(self, zis, zjs, soft_labels=None):
        device = zis.device
        batch_size = zis.size(0)
        zis_list = zis.split(2, dim=0)
        zjs_list = zjs.split(2, dim=0)

        loss = 0
        for zis, zjs in zip(zis_list, zjs_list):
            representations = torch.cat([zjs, zis], dim=0)

            similarity_matrix = self.similarity_function(representations.unsqueeze(1), representations.unsqueeze(0))

            # filter out the scores from the positive samples
            l_pos = torch.diag(similarity_matrix, 2)
            r_pos = torch.diag(similarity_matrix, -2)
            positives = torch.cat([l_pos, r_pos]).view(2 * 2, 1)

            mask_samples_from_same_repr = self._get_correlated_mask(2).to(device)
            negatives = similarity_matrix[mask_samples_from_same_repr].view(2 * 2, -1)

            logits = torch.cat((positives, negatives), dim=1)
            logits /= self.temperature

            if soft_labels is None:
                soft_labels = torch.zeros(4, 3).to(device).long()
                soft_labels[:, 0] = 1
            example_loss = self.criterion(logits, soft_labels)
            loss += example_loss

        return loss / (2 * batch_size)
