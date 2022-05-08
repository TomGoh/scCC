import torch
from torch import nn

class MultiSimilarityLoss(nn.Module):

    def __init__(self, args):
        super(MultiSimilarityLoss, self).__init__()
        self.thresh = 0.5
        self.margin = 0.1
        self.args=args
        self.scale_pos = args.MULTI_SIMILARITY_LOSS_SCALE_POS
        self.scale_neg = args.MULTI_SIMILARITY_LOSS_SCALE_NEG

    def forward(self, feats, labels):
        assert feats.size(0) == labels.size(0), \
            f"feats.size(0): {feats.size(0)} is not equal to labels.size(0): {labels.size(0)}"
        batch_size = feats.size(0)
        sim_mat = torch.matmul(feats, torch.t(feats))
        epsilon = 1e-5
        loss = list()

        for i in range(batch_size):
            if i % self.args.NN_COUNT==0:

                pos_pair_ = sim_mat[i][labels == labels[i]]
                pos_pair_ = pos_pair_[pos_pair_ < 1 - epsilon]
                # 可能自己会被过滤
                neg_pair_ = sim_mat[i][labels != labels[i]]

                neg_pair = neg_pair_[neg_pair_ + self.margin > min(pos_pair_)]
                pos_pair = pos_pair_[pos_pair_ - self.margin < max(neg_pair_)]

                if len(neg_pair) < 1 or len(pos_pair) < 1:
                    continue

                # weighting step
                pos_loss = 1.0 / self.scale_pos * torch.log(
                    1 + torch.sum(torch.exp(-self.scale_pos * (pos_pair - self.thresh))))
                neg_loss = 1.0 / self.scale_neg * torch.log(
                    1 + torch.sum(torch.exp(self.scale_neg * (neg_pair - self.thresh))))
                loss.append(pos_loss + neg_loss)

        if len(loss) == 0:
            return torch.zeros([], requires_grad=True)

        loss = sum(loss) / batch_size
        return loss

class MultiSimilarityLoss_Boost(nn.Module):

    def __init__(self, args):
        super(MultiSimilarityLoss_Boost, self).__init__()
        self.thresh = 0.5
        self.margin = 0.1
        self.args=args
        self.scale_pos = args.MULTI_SIMILARITY_LOSS_SCALE_POS
        self.scale_neg = args.MULTI_SIMILARITY_LOSS_SCALE_NEG

    # Generating Mask for samples with same pseudolabel
    # Cover the mask over the simliarity matrix to gain 
    # the positive pairs and negative pairs
    
    def mask_correlate(self,args):
        block=torch.ones((args.NN_COUNT,args.NN_COUNT),dtype=torch.long)
        block=block.unsqueeze(0)
        block=torch.repeat_interleave(block,args.batch_size,dim=0)

        mask = torch.block_diag(*block)
        mask=mask.bool()

        return mask


    def forward(self, feats, labels):
        assert feats.size(0) == labels.size(0), \
            f"feats.size(0): {feats.size(0)} is not equal to labels.size(0): {labels.size(0)}"
        sim_mat = torch.matmul(feats, torch.t(feats))
        mask=self.mask_correlate(self.args)
 
        pos_pair_=sim_mat[mask]
        mask=torch.logical_not(mask)
        neg_pair_=sim_mat[mask]
        # print(torch.mean(pos_pair_))
        # print(torch.mean(neg_pair_))
        # print(torch.max(pos_pair_))
        # print(torch.max(neg_pair_))
        # print(torch.min(pos_pair_))
        # print(torch.min(neg_pair_))
        
        neg_pair = neg_pair_[neg_pair_ + self.margin > torch.min(pos_pair_)]
        pos_pair = pos_pair_[pos_pair_ - self.margin < torch.max(neg_pair_)]

        # weighting step
        pos_loss = 1.0 / self.scale_pos * torch.log(
            1 + torch.sum(torch.exp(-self.scale_pos * (pos_pair - self.thresh))))
        neg_loss = 1.0 / self.scale_neg * torch.log(
            1 + torch.sum(torch.exp(self.scale_neg * (neg_pair - self.thresh))))
        loss=pos_loss+neg_loss

        return loss