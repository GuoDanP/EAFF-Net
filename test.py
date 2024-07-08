import torch
from tqdm import tqdm
from opt import opt
from utils.metrics import evaluate
import datasets
import numpy as np
from torch.utils.data import DataLoader
from utils.comm import generate_model
from utils.metrics import Metrics
from utils.metrics_tga import precision, recall, F2, dice_score, jac_score
from sklearn.metrics import accuracy_score, confusion_matrix
from operator import add
def print_score(metrics_score,num):
    print(metrics_score)
    print(num)
    jaccard = metrics_score[0]/num
    f1 = metrics_score[1]/num
    recall = metrics_score[2]/num
    precision = metrics_score[3]/num
    acc = metrics_score[4]/num
    f2 = metrics_score[5]/num

    print(f"Jaccard: {jaccard:1.4f} - F1: {f1:1.4f} - Recall: {recall:1.4f} - Precision: {precision:1.4f} - Acc: {acc:1.4f} - F2: {f2:1.4f}")

def calculate_metrics(y_true, y_pred):
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()

    y_pred = y_pred > 0.5
    y_pred = y_pred.reshape(-1)
    y_pred = y_pred.astype(np.uint8)

    y_true = y_true > 0.5
    y_true = y_true.reshape(-1)
    y_true = y_true.astype(np.uint8)
    ## Score
    score_jaccard = jac_score(y_true, y_pred)
    score_f1 = dice_score(y_true, y_pred)
    score_recall = recall(y_true, y_pred)
    score_precision = precision(y_true, y_pred)
    score_fbeta = F2(y_true, y_pred)
    score_acc = accuracy_score(y_true, y_pred)

    return [score_jaccard, score_f1, score_recall, score_precision, score_acc, score_fbeta]

def test():
    print('loading data......')
    test_data = getattr(datasets, opt.dataset)(opt.root, opt.test_data_dir, mode='test')
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=opt.num_workers)
    total_batch = int(len(test_data) / 1)
    model = generate_model(opt)

    model.eval()

    # metrics_logger initialization
    metrics = Metrics(['recall', 'specificity', 'precision', 'F1', 'F2',
                       'ACC_overall', 'IoU_poly', 'IoU_bg', 'IoU_mean'])
    metrics_score_1 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    with torch.no_grad():
        bar = tqdm(enumerate(test_dataloader), total=total_batch)
        for i, data in bar:
            img, gt = data['image'], data['label']

            if opt.use_gpu:
                img = img.cuda()
                gt = gt.cuda()

            output = model(img)
            _recall, _specificity, _precision, _F1, _F2, \
            _ACC_overall, _IoU_poly, _IoU_bg, _IoU_mean = evaluate(output, gt)

            outputs=output[0].squeeze().squeeze()
            gts=gt.squeeze().squeeze()
            #print(gts.shape)
            #print(outputs.shape)
            score_1 = calculate_metrics(gts, outputs)
            metrics_score_1 = list(map(add, metrics_score_1, score_1))


            metrics.update(recall= _recall, specificity= _specificity, precision= _precision, 
                            F1= _F1, F2= _F2, ACC_overall= _ACC_overall, IoU_poly= _IoU_poly, 
                            IoU_bg= _IoU_bg, IoU_mean= _IoU_mean
                        )
    print_score(metrics_score_1,total_batch)
    metrics_result = metrics.mean(total_batch)

    print("Test Result:")
    print('recall: %.4f, specificity: %.4f, precision: %.4f, F1: %.4f, F2: %.4f, '
          'ACC_overall: %.4f, IoU_poly: %.4f, IoU_bg: %.4f, IoU_mean: %.4f'
          % (metrics_result['recall'], metrics_result['specificity'], metrics_result['precision'],
             metrics_result['F1'], metrics_result['F2'], metrics_result['ACC_overall'],
             metrics_result['IoU_poly'], metrics_result['IoU_bg'], metrics_result['IoU_mean']))


if __name__ == '__main__':

    if opt.mode == 'test':
        print('--- PolypSeg Test---')
        test()

    print('Done')
