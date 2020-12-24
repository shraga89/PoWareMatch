import torch
import torch.nn as nn


class MultiTaskLossWrapper(nn.Module):
    def __init__(self, task_num):
        super(MultiTaskLossWrapper, self).__init__()
        self.task_num = task_num
        self.log_vars = nn.Parameter(torch.zeros((task_num)))

    def forward(self, preds, y, P, F):
        mse, crossEntropy = nn.MSELoss(), nn.NLLLoss()

        loss1 = crossEntropy(preds[1], y)
        loss2 = mse(preds[2], P)
        loss3 = mse(preds[3], F)

        precision1 = torch.exp(-self.log_vars[0])
        loss1 = precision1 * loss1 + self.log_vars[0]

        precision2 = torch.exp(-self.log_vars[1])
        loss2 = precision2 * loss2 + self.log_vars[1]

        precision3 = torch.exp(-self.log_vars[2])
        loss3 = precision3 * loss3 + self.log_vars[2]

        return loss1 + loss2 + loss3