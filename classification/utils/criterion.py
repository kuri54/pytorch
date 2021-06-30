# %%
import torch

class LabelSmoothLoss(torch.nn.Module):
    '''Label Smoothing
    Onehot表現のラベルをソフトラベルにして学習させる
    正解ラベル：1 - α
    不正解ラベル：α

    ・過学習に効果あり
    ・ごく微小な精度向上

    Parameters
    ----------
    num_classes: int
    alpha: int
        ラベルに加えるノイズの割合

    Returns
    -------
    loss
    '''
    def __init__(self, num_classes, alpha=0.1):
        super(LabelSmoothLoss, self).__init__()
        self.num_classes = num_classes
        self.softmax = torch.nn.Softmax(dim=1)
        self.alpha = alpha
        self.device = 'cuda'

    def forward(self, output, target):
        probs = self.softmax(output)
        label_one_hot = torch.nn.functional.one_hot(target, self.num_classes).float().to(self.device)
        label_one_hot = label_one_hot * (1. - self.alpha) + self.alpha / float(self.num_classes)
        loss = torch.sum(-label_one_hot * torch.log(probs), dim=1).mean()
        return loss
