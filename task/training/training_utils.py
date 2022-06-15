import os
import torch

def label_smoothing_loss(pred, gold, trg_pad_idx, smoothing_eps=0.1):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''
    gold = gold.contiguous().view(-1)
    n_class = pred.size(1)

    one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
    one_hot = one_hot * (1 - smoothing_eps) + (1 - one_hot) * smoothing_eps / (n_class - 1)
    log_prb = F.log_softmax(pred, dim=1)

    non_pad_mask = gold.ne(trg_pad_idx)
    loss = -(one_hot * log_prb).sum(dim=1)
    loss = loss.masked_select(non_pad_mask).mean()
    return loss

def model_save_name(args):
    save_path = os.path.join(args.model_save_path, args.task, args.data_name, args.tokenizer)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if args.tokenizer == 'spm':
        save_name_pre = f'checkpoint_src_{args.src_vocab_size}_trg_{args.trg_vocab_size}_v_{args.variational_mode}_p_{args.parallel}.pth.tar'
    else:
        save_name_pre = f'checkpoint_v_{args.variational_mode}_p_{args.parallel}.pth.tar'
    save_file_name = os.path.join(save_path, save_name_pre)
    return save_file_name