import os
import time
import torch
from tqdm import tqdm


class Eval_thread():
    def __init__(self, loader, method, dataset, output_dir, cuda):
        self.loader = loader
        self.method = method
        self.dataset = dataset
        self.cuda = cuda
        self.logfile = os.path.join(output_dir, 'result.txt')

    def run(self):
        start_time = time.time()
        mae = self.Eval_mae()
        mean_f, max_f = self.Eval_fmeasure()
        adp_f = self.adap_fmeasure()
        mean_e, max_e = self.Eval_Emeasure()
        adp_e = self.adap_Emeasure()
        iou = self.Eval_IoU()
        self.LOG(
            '{} dataset with {} method get {:.4f} mae, {:.4f} adap-fmeasure, {:.4f} mean-fmeasure, {:.4f} max-fmeasure,  {:.4f} adap-Emeasure, {:.4f} mean-Emeasure, {:.4f} max-Emeasure, {:.4f} IoU..\n'.format(
                self.dataset, self.method, mae, adp_f, mean_f, max_f, adp_e, mean_e, max_e, iou))
        return '[cost:{:.4f}s]{} dataset with {} method get {:.4f} mae, {:.4f} adap-fmeasure, {:.4f} mean-fmeasure,{:.4f} max-fmeasure, {:.4f} adap-Emeasure, {:.4f} mean-Emeasure, {:.4f} max-Emeasure, {:.4f} IoU..'.format(
            time.time() - start_time, self.dataset, self.method, mae, adp_f, mean_f, max_f, adp_e, mean_e, max_e, iou)

    def Eval_mae(self):
        print('eval[MAE]:{} dataset with {} method.'.format(self.dataset, self.method))
        avg_mae, img_num = 0.0, 0.0
        with torch.no_grad():
            for pred, gt in tqdm(self.loader):
                if self.cuda:
                    pred = pred.cuda()
                    gt = gt.cuda()
                mea = torch.abs(pred - gt).mean()
                if mea == mea:  # for Nan
                    avg_mae += mea
                    img_num += 1.0
            avg_mae /= img_num
            return avg_mae.item()

    def adap_fmeasure(self):
        print('eval[adaptive FMeasure]:{} dataset with {} method.'.format(self.dataset, self.method))
        beta2 = 0.3
        avg_f, img_num = 0.0, 0.0
        with torch.no_grad():
            for pred, gt in tqdm(self.loader):
                if self.cuda:
                    pred = pred.cuda()
                    gt = gt.cuda()
                th = 2 * pred.mean()
                if th > 1:
                    th = 1
                pred = (pred >= th).float()
                tp = (pred * gt).sum()
                prec, recall = tp / (pred.sum() + 1e-20), tp / (gt.sum() + 1e-20)
                f_score = (1 + beta2) * prec * recall / (beta2 * prec + recall)
                f_score[f_score != f_score] = 0
                avg_f += f_score
                img_num += 1.0
            score = avg_f / img_num
            return score.item()

    def Eval_fmeasure(self):
        print('eval[mean FMeasure and max FMeasure]:{} dataset with {} method.'.format(self.dataset, self.method))
        beta2 = 0.3
        avg_f, img_num = 0.0, 0.0

        with torch.no_grad():
            for pred, gt in tqdm(self.loader):
                if self.cuda:
                    pred = pred.cuda()
                    gt = gt.cuda()
                prec, recall = self._eval_pr(pred, gt, 255)
                f_score = (1 + beta2) * prec * recall / (beta2 * prec + recall)
                f_score[f_score != f_score] = 0  # for Nan
                avg_f += f_score
                img_num += 1.0
                score = avg_f / img_num
            return score.mean().item(), score.max().item()

    def adap_Emeasure(self):
        print('eval[adaptive EMeasure]:{} dataset with {} method.'.format(self.dataset, self.method))
        avg_e, img_num = 0.0, 0.0
        with torch.no_grad():
            for pred, gt in tqdm(self.loader):
                if self.cuda:
                    pred = pred.cuda()
                    gt = gt.cuda()
                th = 2 * pred.mean()
                if th > 1:
                    th = 1
                y_pred_th = (pred >= th).float()
                fm = y_pred_th - y_pred_th.mean()
                gt = gt - gt.mean()
                align_matrix = 2 * gt * fm / (gt * gt + fm * fm + 1e-20)
                enhanced = ((align_matrix + 1) * (align_matrix + 1)) / 4
                avg_e += torch.sum(enhanced) / (gt.numel() - 1 + 1e-20)
                img_num = img_num + 1
            score = avg_e / img_num
            return score.item()

    def Eval_Emeasure(self):
        print('eval[mean EMeasure and max EMeasure]:{} dataset with {} method.'.format(self.dataset, self.method))
        avg_e, img_num = 0.0, 0.0
        with torch.no_grad():
            scores = torch.zeros(255)
            if self.cuda:
                scores = scores.cuda()
            for pred, gt in tqdm(self.loader):
                if self.cuda:
                    pred = pred.cuda()
                    gt = gt.cuda()
                scores += self._eval_e(pred, gt, 255)
                img_num += 1.0
            scores /= img_num
            return scores.mean().item(), scores.max().item()

    def Eval_IoU(self):

        print('eval[IoU]:{} dataset with {} method.'.format(self.dataset, self.method))

        avg_iou, img_num = 0.0, 0.0
        with torch.no_grad():
            scores = torch.zeros(255)
            if self.cuda:
                scores = scores.cuda()
            for pred, gt in tqdm(self.loader):
                if self.cuda:
                    pred = pred.cuda()
                    gt = gt.cuda()
                scores += self._eval_iou(pred, gt, 255)
                img_num += 1
            scores /= img_num
            return scores.mean().item()

    def LOG(self, output):
        with open(self.logfile, 'a') as f:
            f.write(output)

    def _eval_e(self, y_pred, y, num):
        if self.cuda:
            score = torch.zeros(num).cuda()
            thlist = torch.linspace(0, 1 - 1e-10, num).cuda()
        else:
            score = torch.zeros(num)
            thlist = torch.linspace(0, 1 - 1e-10, num)
        for i in range(num):
            y_pred_th = (y_pred >= thlist[i]).float()
            fm = y_pred_th - y_pred_th.mean()
            gt = y - y.mean()
            align_matrix = 2 * gt * fm / (gt * gt + fm * fm + 1e-20)
            enhanced = ((align_matrix + 1) * (align_matrix + 1)) / 4
            score[i] = torch.sum(enhanced) / (y.numel() - 1 + 1e-20)
        return score

    def _eval_pr(self, y_pred, y, num):
        if self.cuda:
            prec, recall = torch.zeros(num).cuda(), torch.zeros(num).cuda()
            thlist = torch.linspace(0, 1 - 1e-10, num).cuda()
        else:
            prec, recall = torch.zeros(num), torch.zeros(num)
            thlist = torch.linspace(0, 1 - 1e-10, num)
        for i in range(num):
            y_temp = (y_pred >= thlist[i]).float()
            tp = (y_temp * y).sum()
            prec[i], recall[i] = tp / (y_temp.sum() + 1e-20), tp / (y.sum() + 1e-20)
        return prec, recall

    def _eval_iou(self, pred, gt, num):
        if self.cuda:
            score = torch.zeros(num).cuda()
            thlist = torch.linspace(0, 1 - 1e-10, num).cuda()
        else:
            score = torch.zeros(num)
            thlist = torch.linspace(0, 1 - 1e-10, num)
        for i in range(num):
            pred_th = (pred >= thlist[i]).float()
            score[i] = torch.sum((pred_th == 1) & (gt == 1)) / torch.sum((pred_th == 1) | (gt == 1))

        return score
