import torch
import numpy as np

from tools.registry import registry
import logging

logger = logging.getLogger(__name__)


# input: B x * x ... x *
# dim: 0 < scalar
# index: B x M
def batched_index_select(input, dim, index):
    views = [input.shape[0]] + [1 if i != dim else -1 for i in range(1, len(input.shape))]
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.view(views).expand(expanse)
    return torch.gather(input, dim, index)


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
        batch_score = scores.sum() / extended_mask.sum()

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
        score = scores.sum() / len(scores)
        loss = loss.sum() / len(loss)
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
        score = scores.sum() / len(scores)
        loss = loss.sum() / len(loss)
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

        total_loss, total_score = sum(total_loss) / len(total_loss), sum(total_score) / len(total_score)

        return total_loss, total_score


class SupConLoss(torch.nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07, formulation="normal"):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.formulation = formulation
        logger.info(f"SCL Loss Formulation: {self.formulation}")

    def set_formulation(self, formulation):
        self.formulation = formulation

    def forward(self, batch_dict):
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

        if registry.use_rephrasings:
            # joint features
            features = torch.cat(
                [
                    batch_dict[0]["contrastive_projection_norm"].unsqueeze(dim=1),
                    batch_dict[1]["contrastive_projection_norm"].unsqueeze(dim=1)
                ],
                dim=1
            )
        else:
            # joint features
            features = batch_dict[0]["contrastive_projection_norm"].unsqueeze(dim=1)

        # targets for the batch is the one with highest score
        # todo: fix this for multi-label setting!
        labels = batch_dict[0]["target"].argmax(dim=-1).view(-1, 1)

        # samples without an answer cannot work as anchor points
        mask_samples = (batch_dict[0]["target"].sum(dim=-1) != 0).int()

        # mask
        pos_mask = None

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and pos_mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and pos_mask is None:
            pos_mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            pos_mask = torch.eq(labels, labels.T).float().to(device)
        else:
            pos_mask = pos_mask.float().to(device)

        # remove samples without gt
        pos_mask = pos_mask * mask_samples

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

        # for numerical stability, doesn't affect any values ahead
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        pos_mask = pos_mask.repeat(anchor_count, contrast_count)

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(pos_mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        # This is just an inverted identity matrix
        # assert logits_mask.cpu() == (torch.eye(logits_mask.shape[0]) == 0).int()
        pos_mask = pos_mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask

        if self.formulation == "custom":
            negs_mask = (pos_mask == 0).int() * logits_mask
            negs_sum = (exp_logits * negs_mask).sum(dim=-1, keepdim=True)
            denominator = negs_sum + exp_logits * pos_mask
            log_prob = logits - torch.log(denominator.sum(1, keepdim=True))
        else:
            assert self.formulation == "normal"
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # import pdb
        # pdb.set_trace()

        # limit the positives
        # todo: try pulling closer positives w/ rephrasings?
        scl_mask_thresh = int(registry.scl_mask_thresh)
        if scl_mask_thresh > 0:
            secondary_mask = torch.zeros_like(pos_mask)
            half_batch_size = int(len(secondary_mask) / 2)
            secondary_mask[:half_batch_size, half_batch_size:] = torch.eye(half_batch_size)
            secondary_mask[half_batch_size:, :half_batch_size] = torch.eye(half_batch_size)
            for idx, row in enumerate(pos_mask):
                nz_inds = row.nonzero().squeeze(-1).tolist()
                if len(nz_inds) > 0:
                    nz_inds = np.random.choice(nz_inds, size=min(scl_mask_thresh, len(nz_inds)), replace=False)
                    secondary_mask[idx][nz_inds] = 1
            pos_mask = pos_mask * secondary_mask

        # re-scaling rephrasings
        scl_mask_rescale_factor = registry.scl_mask_rescale_factor
        if scl_mask_rescale_factor > 0:
            secondary_mask = torch.zeros_like(pos_mask)
            half_batch_size = int(len(secondary_mask) / 2)
            secondary_mask[:half_batch_size, half_batch_size:] = torch.eye(half_batch_size)
            secondary_mask[half_batch_size:, :half_batch_size] = torch.eye(half_batch_size)
            secondary_mask = secondary_mask * scl_mask_rescale_factor
            secondary_mask[secondary_mask == 0] = 1
            pos_mask = pos_mask * secondary_mask

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (pos_mask * log_prob).sum(1) / torch.max(pos_mask.sum(1), torch.ones(1).to(pos_mask.device))

        # # calculating score (seems pointless)
        # preds = torch.argmax(exp_logits, 1)
        #
        # if registry.use_rephrasings:
        #     scores = (preds == labels.squeeze().repeat(2)).float()
        # else:
        #     scores = (preds == labels.squeeze()).float()
        # scores = scores * pos_mask
        # batch_score = scores.sum() / pos_mask.sum()

        # loss


        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss, -1


def MSELoss(batch_dict):
    mse_loss = torch.nn.MSELoss(reduction="sum")
    total_loss = 0

    common_inds = batch_dict[0]["common_inds"]
    common_inds = common_inds.unsqueeze(1).repeat(1, 12, 1).view(-1, 23)
    common_inds_mask = (common_inds > 0).float().unsqueeze(-1)

    pos_common_inds = batch_dict[1]["common_inds"]
    pos_common_inds = pos_common_inds.unsqueeze(1).repeat(1, 12, 1).view(-1, 23)
    pos_common_inds_mask = (pos_common_inds > 0).float().unsqueeze(-1)

    for layer_attn1, layer_attn2, use_layer in zip(batch_dict[0]["attention_weights"],
                                                   batch_dict[1]["attention_weights"],
                                                   registry.squint_layers):

        que_obj_attn1, que_obj_attn2 = layer_attn1[:, :, :23, -101:], layer_attn2[:, :, :23, -101:]

        if registry.squint_type == "common":
            common_attn1, common_attn2 = batched_index_select(que_obj_attn1.view(-1, 23, 101), 1, common_inds), \
                                         batched_index_select(que_obj_attn2.view(-1, 23, 101), 1, pos_common_inds)
            common_attn1, common_attn2 = common_attn1 * common_inds_mask, common_attn2 * pos_common_inds_mask
        elif registry.squint_type == "average":
            attn_mask1, attn_mask2 = batch_dict[0]["question_mask"].unsqueeze(dim=-2).unsqueeze(dim=-1), batch_dict[1]["question_mask"].unsqueeze(dim=-2).unsqueeze(dim=-1)
            common_attn1, common_attn2 = que_obj_attn1 * attn_mask1, que_obj_attn2 * attn_mask2
            common_attn1, common_attn2 = common_attn1.sum(dim=-2), common_attn2.sum(dim=-2)

        else:
            raise ValueError

        if use_layer:
            loss = mse_loss(common_attn1.view(-1, 101), common_attn2.view(-1, 101))
            total_loss += loss

    return total_loss
