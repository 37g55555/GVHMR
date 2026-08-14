import torch
import torch.nn.functional as F
import math
from einops import rearrange


def uniform(shape, device=None):
    return torch.zeros(shape, device=device).float().uniform_(0, 1)


def get_mask_subset_prob(mask, prob):
    subset_mask = torch.bernoulli(mask, p=prob) & mask
    return subset_mask


def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))


def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


def gumbel_sample(t, temperature = 1., dim = 1):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim=dim)


def top_k(logits, k = 1, thres = 0.9, dim = 1):
    if k is None:
        k = math.ceil((1 - thres) * logits.shape[dim])
    val, ind = logits.topk(k, dim = dim)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(dim, ind, val)
    # func verified
    # print(probs)
    # print(logits)
    # raise
    return probs


def cosine_schedule(t):
    return torch.cos(t * math.pi * 0.5)


def cal_performance(pred, labels, ignore_index=None, smoothing=0., tk=1, focal_gamma=0.):
    loss = cal_loss(pred, labels, ignore_index, smoothing=smoothing, focal_gamma=focal_gamma)
    # pred_id = torch.argmax(pred, dim=1)
    # mask = labels.ne(ignore_index)
    # n_correct = pred_id.eq(labels).masked_select(mask)
    # acc = torch.mean(n_correct.float()).item()
    pred_id_k = torch.topk(pred, k=tk, dim=1).indices
    pred_id = pred_id_k[:, 0]
    mask = labels.ne(ignore_index)
    n_correct = (pred_id_k == labels.unsqueeze(1)).any(dim=1).masked_select(mask)
    acc = torch.mean(n_correct.float()).item()

    return loss, pred_id, acc


def cal_loss(pred, labels, ignore_index=None, smoothing=0., focal_gamma=0.):
    '''Calculate cross entropy loss, apply label smoothing if needed.'''
    # print(pred.shape, labels.shape) #torch.Size([64, 1028, 55]) torch.Size([64, 55])
    # print(pred.shape, labels.shape) #torch.Size([64, 1027, 55]) torch.Size([64, 55])
    if smoothing:
        space = 2
        n_class = pred.size(1)
        mask = labels.ne(ignore_index)
        one_hot = rearrange(F.one_hot(labels, n_class + space), 'a ... b -> a b ...')[:, :n_class]
        # one_hot = torch.zeros_like(pred).scatter(1, labels.unsqueeze(1), 1)
        sm_one_hot = one_hot * (1 - smoothing) + (1 - one_hot) * smoothing / (n_class - 1)
        neg_log_prb = -F.log_softmax(pred, dim=1)
        loss = (sm_one_hot * neg_log_prb).sum(dim=1)
        # loss = F.cross_entropy(pred, sm_one_hot, reduction='none')
        loss = torch.mean(loss.masked_select(mask))
    else:
        if focal_gamma > 0:
            # Focal loss
            ce_loss = F.cross_entropy(pred, labels, reduction='none', ignore_index=ignore_index)
            ce_loss = ce_loss[ce_loss > 0.] 
            pt = torch.exp(-ce_loss)
            loss = ((1 - pt) ** focal_gamma * ce_loss).mean()
        else:
            loss = F.cross_entropy(pred, labels, ignore_index=ignore_index)

    return loss

