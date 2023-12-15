import numpy as np
import pandas as pd


def evaluate(prediction, ground_truth, mask, report=False):
    assert ground_truth.shape == prediction.shape, 'shape mis-match'
    performance = {}
    # mse
    performance['mse'] = np.linalg.norm((prediction - ground_truth) * mask) ** 2 / np.sum(mask)
    # IC
    df_pred = pd.DataFrame(prediction * mask)
    df_gt = pd.DataFrame(ground_truth * mask)

    ic = []
    mrr_top = 0.0
    all_miss_days_top = 0
    bt_long = 1.0
    bt_long5 = 1.0
    bt_long10 = 1.0
    irr = 0.0
    sharpe_li5 = []
    prec_10 = []

    for i in range(prediction.shape[1]):
        # IC
        ic.append(df_pred[i].corr(df_gt[i]))

        rank_gt = np.argsort(ground_truth[:, i])
        gt_top1 = set()
        gt_top5 = set()
        gt_top10 = set()

        for j in range(1, prediction.shape[0] + 1):
            cur_rank = rank_gt[-1 * j]
            if mask[cur_rank][i] < 0.5:
                continue
            if len(gt_top1) < 1:
                gt_top1.add(cur_rank)
            if len(gt_top5) < 5:
                gt_top5.add(cur_rank)
            if len(gt_top10) < 10:
                gt_top10.add(cur_rank)

        rank_pre = np.argsort(prediction[:, i])
        pre_top1 = set()
        pre_top5 = set()
        pre_top10 = set()
        for j in range(1, prediction.shape[0] + 1):
            cur_rank = rank_pre[-1 * j]
            if mask[cur_rank][i] < 0.5:
                continue
            if len(pre_top1) < 1:
                pre_top1.add(cur_rank)
            if len(pre_top5) < 5:
                pre_top5.add(cur_rank)
            if len(pre_top10) < 10:
                pre_top10.add(cur_rank)

        top1_pos_in_gt = 0
        for j in range(1, prediction.shape[0] + 1):
            cur_rank = rank_gt[-1 * j]
            if mask[cur_rank][i] < 0.5:
                continue
            else:
                top1_pos_in_gt += 1
                if cur_rank in pre_top1:
                    break
        if top1_pos_in_gt == 0:
            all_miss_days_top += 1
        else:
            mrr_top += 1.0 / top1_pos_in_gt

        real_ret_rat_top = ground_truth[list(pre_top1)[0]][i]
        bt_long += real_ret_rat_top
        gt_irr = 0.0

        for gt in gt_top10:
            gt_irr += ground_truth[gt][i]

        real_ret_rat_top5 = 0
        for pre in pre_top5:
            real_ret_rat_top5 += ground_truth[pre][i]
        irr += real_ret_rat_top5
        real_ret_rat_top5 /= 5
        bt_long5 += real_ret_rat_top5

        prec = 0.0
        real_ret_rat_top10 = 0
        for pre in pre_top10:
            real_ret_rat_top10 += ground_truth[pre][i]
            prec += (ground_truth[pre][i] >= 0)
        prec_10.append(prec / 10)
        real_ret_rat_top10 /= 10
        bt_long10 += real_ret_rat_top10
        sharpe_li5.append(real_ret_rat_top5)

    performance['IC'] = np.mean(ic)
    performance['RIC'] = np.mean(ic) / np.std(ic)
    sharpe_li5 = np.array(sharpe_li5)
    performance['sharpe5'] = (np.mean(sharpe_li5)/np.std(sharpe_li5))*15.87
    performance['prec_10'] = np.mean(prec_10)
    return performance



