import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseLoss(nn.Module):
    def __init__(self):
        super(BaseLoss, self).__init__()

    def forward(self, preds, targets, weight=None):
        N = len(preds)
        if weight is None:
            weight = preds[0].new_ones(1)
        errs = [self._forward(preds[n], targets[n], weight)
                for n in range(N)]
        err = torch.mean(torch.stack(errs))
    
        return err

class SMOOTHL1Loss(BaseLoss):
    def __init__(self):
        super(L1Loss, self).__init__()

    def _forward(self, pred, target, weight):
        # return torch.mean(weight * torch.abs(pred - target))
        return F.smooth_l1_loss(pred, target)


class L1Loss(BaseLoss):
    def __init__(self):
        super(L1Loss, self).__init__()

    def _forward(self, pred, target, weight):
        return torch.mean(weight * torch.abs(pred - target))


class L2Loss(BaseLoss):
    def __init__(self):
        super(L2Loss, self).__init__()

    def _forward(self, pred, target, weight):
        return torch.mean(weight * torch.pow(pred - target, 2))


class MSELoss(BaseLoss):
    def __init__(self):
        super(MSELoss, self).__init__()

    def _forward(self, pred, target, weight):
        return F.mse_loss(pred, target)


class BCELoss(BaseLoss):
    def __init__(self):
        super(BCELoss, self).__init__()

    def _forward(self, pred, target, weight):
        return F.binary_cross_entropy(pred, target, weight=weight)


from itertools import permutations

class UPITLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def sisnr(x, s, eps=1e-8):
        """
        calculate training loss
        input:
            x: separated signal, N x S tensor
            s: reference signal, N x S tensor
        Return:
            sisnr: N tensor
        """

        def l2norm(mat, keepdim=False):
            return torch.norm(mat, dim=-1, keepdim=keepdim)

        if x.shape != s.shape:
            raise RuntimeError(
                "Dimention mismatch when calculate si-snr, {} vs {}".format(
                    x.shape, s.shape))
        x_zm = x - torch.mean(x, dim=-1, keepdim=True)
        s_zm = s - torch.mean(s, dim=-1, keepdim=True)
        t = torch.sum(
            x_zm * s_zm, dim=-1,
            keepdim=True) * s_zm / (l2norm(s_zm, keepdim=True)**2 + eps)

        return 20 * torch.log10(eps + l2norm(t) / (l2norm(x_zm - t) + eps))


    def forward(self, ests, refs):
        # spks x batch x S
        num_spks = len(refs)

        def sisnr_loss(permute):
            # for one permute
            return sum(
                [self.sisnr(ests[s], refs[t])
                for s, t in enumerate(permute)]) / len(permute)
                # average the value

        # P x N
        N = refs[0].size(0)
        sisnr_mat = torch.stack(
            [sisnr_loss(p) for p in permutations(range(num_spks))])
        max_perutt, _ = torch.max(sisnr_mat, dim=0)
        import numpy as np
        if torch.isnan(-torch.sum(max_perutt) / N):
            print(N, max_perutt, sisnr_mat)
            raise ValueError

        # si-snr
        return -torch.sum(max_perutt) / N

class SISNRLoss(BaseLoss):
    def __init__(self):
        super(SISNRLoss, self).__init__()
        self.eps = 1e-8

    def _forward(self, pred, target, weight=None):
        def l2norm(mat, keepdim=False):
            return torch.norm(mat, dim=-1, keepdim=keepdim)

        if pred.shape != target.shape:
            raise RuntimeError(
                "Dimention mismatch when calculate si-snr, {} vs {}".format(
                    pred.shape, target.shape))

        pred_zm = pred - torch.mean(pred, dim=-1, keepdim=True)
        target_zm = target - torch.mean(target, dim=-1, keepdim=True)
        t = torch.sum(
            pred_zm * target_zm, dim=-1,
            keepdim=True) * target_zm / (l2norm(target_zm, keepdim=True)**2 + self.eps)

        return 20 * torch.log10(self.eps + l2norm(t) / (l2norm(pred_zm - t) + self.eps))