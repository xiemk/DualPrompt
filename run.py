
import os
import sys
import random
import json
import time
import numpy as np
import argparse
import torch
import clip


from src_files.data.data import get_datasets
from src_files.utils.helper import get_raw_dict, function_mAP
from src_files.utils.logger import setup_logger
from src_files.utils.meter import AverageMeter, ProgressMeter

NUM_CLASS = {'voc2007': 20, 'voc2012': 20, 'coco2014': 80, 'vg256': 256, 'objects365':365}
MODEL_NAME = {'RN50':'RN50', 'RN101':'RN101', 'RN50x4':'RN50x4', 'RN50x16':'RN50x16', 'RN50x64':'RN50x64', 'ViT-B/32':'ViT-B-32', 'ViT-B/16':'ViT-B-16', 'ViT-L/14':'ViT-L-14', 'ViT-L/14@336px':'ViT-L-14-336px'}

def get_args():
    parser = argparse.ArgumentParser(description='DualPrompt: training-free zero-shot multi-label classification with CLIP.')

    # data
    parser.add_argument('--data_name', help='dataset name', default='coco2014', choices=['coco2014', 'nus','vg256','objects365'])
    parser.add_argument('--data_dir', help='root directory containing dataset folders', default='./data')
    parser.add_argument('--image_size', default=224, type=int,
                        help='size of input images')
    parser.add_argument('--output', metavar='DIR', default='./outputs',
                        help='path to output folder')
    parser.add_argument('-b', '--batch_size', default=64, type=int,
                        help='batch size')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('--model_name', default='RN101', choices=['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px'])
    parser.add_argument('--lam', default=0.5, type=float)
    parser.add_argument('--coo_derive', default='data', type=str)
    parser.add_argument('--coo_mode', default='both', choices=['top', 'bottom', 'both'])
    parser.add_argument('--coo_thre', default= 0.2, type = float)
    parser.add_argument('--coo_max_n', default= 4, type = int)
    parser.add_argument('--sample_ratio', default= 0.01, type = float)
    parser.add_argument('--n_grid', default=3, type=int)
    parser.add_argument('--seed', default=1, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=0, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('-p', '--print_freq', default=50, type=int, metavar='N', help='print frequency')

    args = parser.parse_args()

    args.num_classes = NUM_CLASS[args.data_name]
    args.data_dir = os.path.join(args.data_dir, args.data_name) 
    
    if args.coo_derive == 'chatgpt':
        args.output = os.path.join(args.output, args.data_name, f'pat_dual_prompt_{args.data_name}_{MODEL_NAME[args.model_name]}_{args.image_size}_{args.coo_derive}_{args.sample_ratio}_{args.seed}')
    elif args.coo_derive == 'data':
        args.output = os.path.join(args.output, args.data_name, f'pat_dual_prompt_{args.data_name}_{MODEL_NAME[args.model_name]}_{args.image_size}_{args.lam}_{args.coo_derive}_{args.sample_ratio}_{args.coo_mode}_{args.coo_thre}_{args.coo_max_n}_{args.seed}')
    elif args.coo_derive == 'prior':
        args.output = os.path.join(args.output, args.data_name, f'pat_dual_prompt_{args.data_name}_{MODEL_NAME[args.model_name]}_{args.image_size}_{args.coo_derive}_{args.seed}')

    return args

def main():
    args = get_args()
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
    
    os.makedirs(args.output, exist_ok=True)

    logger = setup_logger(output=args.output, color=False, name="XXX")
    logger.info("Command: "+' '.join(sys.argv))

    path = os.path.join(args.output, "config.json")
    with open(path, 'w') as f:
        json.dump(get_raw_dict(args), f, indent=2)
    logger.info("Full config saved to {}".format(path))

    return main_worker(args, logger)

def main_worker(args, logger):

    logger.info('Creating model...')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load(args.model_name, device)

    train_dataset, val_dataset = get_datasets(args, True)
    logger.info("len(train_dataset)): {}".format(len(train_dataset)))
    logger.info("len(val_dataset)): {}".format(len(val_dataset)))

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)

    cls_texts = load_cls_names(f'./src_files/{args.data_name}_cls_names.txt')
    if args.coo_derive == 'chatgpt':
        n_sample = int(train_dataset.Y.shape[0]*args.sample_ratio)
        train_labels = train_dataset.Y[np.random.choice(train_dataset.Y.shape[0], n_sample, replace=False), :]
        pos_prop = train_labels.sum(0)/n_sample
        with open(os.path.join(os.path.dirname(__file__), 'chatgpt', f'{args.data_name}_coo_chatgpt.json'), 'r') as file:
            coo_cls_texts = json.load(file)
    elif args.coo_derive == 'prior':
        train_labels = train_dataset.Y
        pos_prop = train_labels.sum(0)/train_labels.shape[0]
        a = (train_labels.T @ train_labels)
        b = train_labels.sum(0)[:, np.newaxis]
        coo_probs = np.divide(a, b, where=b != 0, out=np.zeros_like(a))
        np.fill_diagonal(coo_probs, 0)
        coo_cls_texts = get_coo_cls(cls_texts, coo_probs, args.coo_mode, args.coo_thre, args.coo_max_n)
    elif args.coo_derive == 'data':
        n_sample = int(train_dataset.Y.shape[0]*args.sample_ratio)
        train_labels = train_dataset.Y[np.random.choice(train_dataset.Y.shape[0], n_sample, replace=False), :]
        pos_prop = train_labels.sum(0)/n_sample
        a = (train_labels.T @ train_labels)
        b = train_labels.sum(0)[:, np.newaxis]
        coo_probs = np.divide(a, b, where=b != 0, out=np.zeros_like(a))
        np.fill_diagonal(coo_probs, 0)
        coo_cls_texts = get_coo_cls(cls_texts, coo_probs, args.coo_mode, args.coo_thre, args.coo_max_n)
        with open(os.path.join(args.output, 'coo_dict.json'), "w", encoding="utf-8") as f:
            json.dump(coo_cls_texts, f, ensure_ascii=False)
            

    disc_text_features, corr_text_features = generate_dual_prompt(model, cls_texts, coo_cls_texts, device)
    

    batch_time = AverageMeter('Time', ':5.3f')
    mem = AverageMeter('Mem', ':.0f', val_only=True)

    progress = ProgressMeter(len(val_loader), [batch_time, mem], prefix='Val:')

    labels = []
    logits = {'dual': []}
    preds = {'dual': []}
    end = time.time()
    # model.eval()
    for i, (img_inputs, targets) in enumerate(val_loader):

        batch_size = targets.shape[0]
        img_inputs = torch.cat(img_inputs, dim=0).to(device)


        with torch.no_grad():

            img_features = model.encode_image(img_inputs)
        
            img_features /= img_features.norm(dim=-1, keepdim=True)

            logit_scale = model.logit_scale.exp()

            disc_logit = img_features @ disc_text_features.t()
            

            img_disc_logit = disc_logit[:batch_size]
            pat_disc_logit = disc_logit[batch_size:]
            pat_disc_logit = logit_attention(batch_size, pat_disc_logit, pat_disc_logit)

            dual_disc_logit = (img_disc_logit + pat_disc_logit)/2

            corr_logit = img_features @ corr_text_features.t()
            img_corr_logit = corr_logit[:batch_size]
            pat_corr_logit = corr_logit[batch_size:]
            pat_corr_logit = logit_attention(batch_size, pat_corr_logit, pat_corr_logit)

            dual_corr_logit = (img_corr_logit+pat_corr_logit)/2

            img_dual_logit = (img_disc_logit+img_corr_logit)/2

            dual_logit = (1-args.lam)/2* (img_disc_logit+pat_disc_logit) + args.lam/2*(img_corr_logit+pat_corr_logit)

            img_disc_logit = logit_scale * img_disc_logit
            pat_disc_logit = logit_scale * pat_disc_logit
            img_corr_logit = logit_scale * img_corr_logit
            pat_corr_logit = logit_scale * pat_corr_logit
            dual_disc_logit = logit_scale * dual_disc_logit
            dual_corr_logit = logit_scale * dual_corr_logit
            img_dual_logit = logit_scale * img_dual_logit
            dual_logit = logit_scale * dual_logit

        labels.append(targets)
        logits['dual'].append(dual_logit.detach().cpu())
        preds['dual'].append(dual_logit.softmax(dim=-1).detach().cpu())

        mem.update(torch.cuda.max_memory_allocated() / 1024.0 / 1024.0)
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i, logger)

    
    labels = torch.cat(labels).numpy()
    probs = torch.cat(preds['dual']).numpy()
    dual_logits = torch.cat(logits['dual']).numpy()

    mAP, APs = function_mAP(labels, probs)
    n_pos = (labels.shape[0] * pos_prop).astype(int)
    pred_labels = label_decision(probs, n_pos)
    CP, CR, CF1, OP, OR, OF1 = calculate_metric(pred_labels, labels)

    logger.info('===== Metrics =====')
    logger.info(
        f"mAP: {mAP:.2f} | CP: {CP:.2f} | CR: {CR:.2f} | CF1: {CF1:.2f} | "
        f"OP: {OP:.2f} | OR: {OR:.2f} | OF1: {OF1:.2f}"
    )

    np.save(os.path.join(args.output, 'dual_logit.npy'), dual_logits)
    np.save(os.path.join(args.output, 'dual_prob.npy'), probs)
    np.save(os.path.join(args.output, 'APs.npy'), APs)

    summary_metrics = {
        'mAP': float(mAP),
        'CP': float(CP),
        'CR': float(CR),
        'CF1': float(CF1),
        'OP': float(OP),
        'OR': float(OR),
        'OF1': float(OF1)
    }
    with open(os.path.join(args.output, 'summary_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(summary_metrics, f, indent=2)

    np.save(os.path.join(args.output, 'mAP.npy'), mAP)
    np.save(os.path.join(args.output, 'CF1.npy'), CF1)
    np.save(os.path.join(args.output, 'OF1.npy'), OF1)
    
    return 0

def compute_AP(predictions, labels):
    num_class = predictions.size(1)
    ap = torch.zeros(num_class).to(predictions.device)
    empty_class = 0
    for idx_cls in range(num_class):
        prediction = predictions[:, idx_cls]
        label = labels[:, idx_cls]

        if (label > 0).sum() == 0:
            empty_class += 1
            continue
        binary_label = torch.clamp(label, min=0, max=1)
        sorted_pred, sort_idx = prediction.sort(descending=True)
        sorted_label = binary_label[sort_idx]
        tmp = (sorted_label == 1).float()
        tp = tmp.cumsum(0)
        fp = (sorted_label != 1).float().cumsum(0)
        num_pos = binary_label.sum()
        rec = tp/num_pos
        prec = tp/(tp+fp)
        ap_cls = (tmp*prec).sum()/num_pos
        ap[idx_cls].copy_(ap_cls)
    return ap

def label_decision(logits, n_pos):
    sorted_logits = -np.sort(-logits, axis=0)

    thresholds = sorted_logits[n_pos, np.arange(sorted_logits.shape[1])][np.newaxis, :]

    preds = np.zeros(logits.shape)

    preds[logits>thresholds] = 1

    return preds


def calculate_metric(preds, labels):

    n_correct_pos = (labels*preds).sum(0)
    n_pred_pos = ((preds==1)).sum(0)
    n_true_pos = labels.sum(0)
    OP = n_correct_pos.sum()/n_pred_pos.sum()
    CP = np.nanmean(n_correct_pos/n_pred_pos)
    OR = n_correct_pos.sum()/n_true_pos.sum()
    CR = np.nanmean(n_correct_pos/n_true_pos)

    CF1 = (2 * CP * CR) / (CP + CR)
    OF1 = (2 * OP * OR) / (OP + OR)

    return CP*100, CR*100, CF1*100, OP*100, OR*100, OF1*100


def get_coo_cls(cls_name, coo_probs, coo_mode, coo_thre, coo_max_n):

    n_cls = coo_probs.shape[0]
    coo_cls = {}
    for i in range(n_cls):
        if coo_mode == 'top':
            coo_prob = coo_probs[i,:]
            coo_cls[cls_name[i]] = [cls_name[j] for j in np.argsort(-coo_prob)[:coo_max_n] if coo_prob[j]>coo_thre]
        elif coo_mode == 'bottom':
            coo_prob = coo_probs[:,i]
            coo_cls[cls_name[i]] = [cls_name[j] for j in np.argsort(-coo_prob)[:coo_max_n] if coo_prob[j]>coo_thre]
        elif coo_mode == 'both':
            coo_prob_top = coo_probs[i,:]
            coo_prob_bottom = coo_probs[:,i]
            top_classes = [cls_name[j] for j in np.argsort(-coo_prob_top)[:coo_max_n] if coo_prob_top[j]>coo_thre]
            bottom_classes = [cls_name[j] for j in np.argsort(-coo_prob_bottom)[:coo_max_n] if coo_prob_bottom[j]>coo_thre]
            coo_cls[cls_name[i]] = list(set(top_classes+bottom_classes))
    
    return coo_cls
    
def generate_dual_prompt(model, cls_texts, coo_cls_texts, device):

    with torch.no_grad():
        disc_text_features = clip.encode_text_with_disc_prompt_ensemble(model, cls_texts, device)
        corr_text_features = clip.encode_text_with_corr_prompt_ensemble(model, cls_texts, coo_cls_texts, device)
        disc_text_features = disc_text_features / disc_text_features.norm(dim=-1, keepdim=True)
        corr_text_features = corr_text_features / corr_text_features.norm(dim=-1, keepdim=True)
    
    return disc_text_features, corr_text_features

def logit_attention(batch_size, logits_pat_1, logits_pat_2):

    split_list1 = torch.split(logits_pat_1, batch_size)          # [64,80] -> 4 * [16,80]
    logits_joint1 = torch.stack(split_list1, dim=1)                  # 4 * [16,80] -> [16, 4, 80]
    logits_sfmx1 = torch.softmax(logits_joint1, dim=1)               # [16, {4}, 80]
    
    split_list2 = torch.split(logits_pat_2, batch_size)          # [64,80] -> 4 * [16,80]
    logits_joint2 = torch.stack(split_list2, dim=1)                  # 4 * [16,80] -> [16, 4, 80]
    
    logits_joint = (logits_sfmx1 * logits_joint2).sum(dim=1)          # [16, 4, 80] -> [16,80]
    return logits_joint

def load_cls_names(file_path):

    cls_names = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            cls_names.append(line.strip())

    return cls_names


if __name__ == '__main__':
    main()