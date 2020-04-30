import torch
import numpy as np

from tools.registry import registry


class NTXentLoss(torch.nn.Module):

    def __init__(self, batch_size):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = registry.loss_params.temperature
        # self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(registry.loss_params.use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="none")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask
        # return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs, mask):
        """
        mask: For samples that don't have positives, we mask the entire row!
        """
        representations = torch.cat([zjs, zis], dim=0)
        similarity_matrix = self.similarity_function(representations, representations)

        try:
            # filter out the scores from the positive samples
            l_pos = torch.diag(similarity_matrix, self.batch_size)
            r_pos = torch.diag(similarity_matrix, -self.batch_size)
            positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

            negatives = similarity_matrix[self.mask_samples_from_same_repr.to(zis.device)].view(2 * self.batch_size, -1)
        except:
            import pdb
            pdb.set_trace()

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * self.batch_size).to(zis.device).long()
        loss = self.criterion(logits, labels)

        # mask helps to fix issues with image-negatives (that do not have rephrasings)
        assert len(mask) == self.batch_size
        extended_mask = torch.cat([mask, mask], dim=0)
        loss = loss * extended_mask
        loss = loss.sum()

        # calculating score
        preds = torch.argmax(logits, 1)
        scores = (preds == 0).float()
        scores = scores * extended_mask
        batch_score = scores.sum()/extended_mask.sum()

        return loss / (2 * self.batch_size), batch_score


class SCLLoss(torch.nn.Module):

    def __init__(self):
        super(SCLLoss, self).__init__()
        self.temperature = registry.loss_params.temperature
        self.use_second = registry.loss_params.get("use_second", False)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.similarity_function = self._get_similarity_function(registry.loss_params.use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="none")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity


    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def row_loss_score(self, row_idx, row, row_mask):
        try:
            row_mask = row_mask.bool()
            row_negatives = torch.masked_select(row, row_mask)
            # select positives
            positives_mask = (row_mask != True)
            positives_mask[row_idx] = False
            logits_positives = torch.masked_select(row, positives_mask).view(-1, 1)
        except:
            import pdb
            pdb.set_trace()

        # calculate loss
        logits_negatives = row_negatives.repeat(len(logits_positives), 1)
        logits = torch.cat([logits_positives, logits_negatives], dim=1)
        targets = torch.zeros(len(logits), dtype=torch.long).to(logits.device)
        try:
            loss = self.criterion(logits, targets)
        except:
            import pdb
            pdb.set_trace()
        # calculate score
        preds = torch.argmax(logits, dim=1)
        scores = (preds == 0).float()
        score = scores.sum()/len(scores)
        loss = loss.sum()/len(loss)
        return loss, score

    def row_loss_score_second(self, row_idx, row, row_mask):
        try:
            row_mask = row_mask.bool()
            row_negatives = torch.masked_select(row, row_mask)
            # select positives
            positives_mask = (row_mask != True)
            positives_mask[row_idx] = False
            logits_positives = torch.masked_select(row, positives_mask).view(-1, 1)
        except:
            import pdb
            pdb.set_trace()

        # calculate loss
        logits_negatives = row_negatives.repeat(len(logits_positives), 1)
        logits_positives = logits_positives.repeat(1, len(logits_positives))
        logits = torch.cat([logits_positives, logits_negatives], dim=1)
        # targets are places where we have positives
        targets = torch.arange(len(logits), dtype=torch.long).to(logits.device)
        try:
            loss = self.criterion(logits, targets)
        except:
            import pdb
            pdb.set_trace()
        # calculate score
        preds = torch.argmax(logits, dim=1)
        scores = (preds == targets).float()
        score = scores.sum()/len(scores)
        loss = loss.sum()/len(loss)
        return loss, score



    def forward(self, zis, zjs, scl_mask):
        """
        mask: For samples that don't have positives, we mask the entire row!
        """
        representations = torch.cat([zjs, zis], dim=0)
        similarity_matrix = self.similarity_function(representations, representations)

        total_loss = []
        total_score = []

        # todo: can be optimized by clubbing rows with same answer together
        for row_idx, (row, row_mask) in enumerate(zip(similarity_matrix, scl_mask)):
            if not self.use_second:
                row_loss, row_score = self.row_loss_score(row_idx, row, row_mask)
            else:
                row_loss, row_score = self.row_loss_score_second(row_idx, row, row_mask)

            total_loss.append(row_loss)
            total_score.append(row_score)

        total_loss, total_score = sum(total_loss)/len(total_loss), sum(total_score)/len(total_score)

        return total_loss, total_score




























