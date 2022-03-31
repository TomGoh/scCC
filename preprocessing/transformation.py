import numpy as np
import torch


def chiasma(original, prob=1, percentage=0.8):
    if original is not None:
        geneCount = original.shape[1]
        s = np.random.uniform(0, 1)
        if s < prob:
            chiasma_instance = int(geneCount * percentage / 2)
            chiasma_pair = np.random.randint(geneCount, size=(chiasma_instance, 2))
            # print(chiasma_pair)
            copy = original.clone()
            copy[:, chiasma_pair[:, 0]], copy[:, chiasma_pair[:, 1]] = copy[:, chiasma_pair[:, 1]], copy[:,
                                                                                                    chiasma_pair[:, 0]]
            return copy
        else:
            return original


def random_mask(original, prob=0.8, percentage=0.1):
    if original is not None:
        cellCount, geneCount = original.shape
        s = np.random.uniform(0, 1)
        # print(s)
        if s < prob:
            mask = np.concatenate([np.ones(int(geneCount * percentage), dtype=bool),
                                   np.zeros(geneCount - int(geneCount * percentage), dtype=bool)])
            np.random.shuffle(mask)
            copy = original.clone()
            copy[:, mask] = 0
            return copy


def gaussian_noise(original, prob=0.8):
    if original is not None:
        s=np.random.uniform(0,1)
        if s<prob:
            cellShape = original.shape
            noise = 0.1 * torch.randn(size=cellShape)
            copy = original.clone()+noise
            return copy


def transformation(original):
    copy=torch.ones(size=original.shape)
    
    # print(copy.shape)
    # for index,singleCell in enumerate(copy):
    #     tmp=gaussian_noise(singleCell)
        # if tmp is not None:
        #     copy[index]=tmp
        # else:
        #     copy[index]=singleCell
    
    # copy=chiasma(copy)
    # copy = gaussian_noise(original)
    # if copy is not None:
    #     copy = chiasma(copy)
    # else:
    #     copy = chiasma(original)
    # if copy is not None:
    #     copy = random_mask(copy)
    # else:
    #     copy = random_mask(original)
    return copy

