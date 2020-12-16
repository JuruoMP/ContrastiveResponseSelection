import numpy as np
import torch
import torch.nn as nn


class NTXentLoss(nn.Module):
    def __init__(self, temperature, use_cosine_similarity):
        super(NTXentLoss, self).__init__()
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
