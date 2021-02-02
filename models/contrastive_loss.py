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
    def criterion(self, logits, soft_labels, reduction='mean'):
        logits = torch.exp(logits)
        ps = logits / logits.sum(dim=1, keepdim=True).expand(logits.size())
        loss = -(torch.log(ps) * soft_labels)
        if reduction == 'sum':
            loss = loss.sum()
        else:
            loss = loss.sum(dim=1).mean()
        return loss

    def forward(self, zis, zjs, batch_soft_logits=None):
        device = zis.device
        batch_size = zis.size(0)
        zis_list = zis.split(2, dim=0)
        zjs_list = zjs.split(2, dim=0)
        if batch_soft_logits is None:
            batch_soft_logits = [None for _ in range(len(zis_list))]

        losses = []
        for zis, zjs, soft_logits in zip(zis_list, zjs_list, batch_soft_logits):
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

            if soft_logits is None:
                example_loss = torch.nn.functional.cross_entropy(logits, torch.LongTensor([0, 0, 0, 0]).to(logits.device), reduction='mean')
            elif len(soft_logits.size()) == 1:
                distance_matrix = torch.abs(soft_logits.unsqueeze(1) - soft_logits.unsqueeze(0))
                distance_matrix_logits = torch.stack([
                    distance_matrix[0, 1:],
                    torch.cat((distance_matrix[1, 0:1], distance_matrix[1, 2:4]), dim=0),
                    torch.cat((distance_matrix[2, 3:4], distance_matrix[2, 0:2]), dim=0),
                    torch.cat((distance_matrix[3, 2:3], distance_matrix[3, 0:2]), dim=0)
                ], dim=0)
                target_distribution = 1 - distance_matrix_logits + 1e-6
                target_distribution = target_distribution / target_distribution.sum(dim=1, keepdim=True)
                example_loss = self.criterion(logits, target_distribution)
            elif len(soft_logits.size()) == 2:
                target_distribution = soft_logits + 1e-6
                target_distribution = target_distribution / target_distribution.sum(dim=1, keepdim=True)
                example_loss = self.criterion(logits, target_distribution)
            else:
                raise Exception('Invalid soft logits')
            losses.append(example_loss)

        return losses


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = torch.device('cuda') if features.is_cuda else torch.device('cpu')

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
