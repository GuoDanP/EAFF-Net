from genericpath import exists
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import datasets
from utils.metrics import evaluate
from opt import opt
from utils.comm import generate_model
from utils.loss import DeepSupervisionLoss,  BceDiceLoss
from utils.metrics import Metrics
import os


def valid(model, valid_dataloader, total_batch):
    

    model.eval()

    # Metrics_logger initialization
    metrics = Metrics(['recall', 'specificity', 'precision', 'F1', 'F2',
                       'ACC_overall', 'IoU_poly', 'IoU_bg', 'IoU_mean'])

    with torch.no_grad():
        # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标.
        # tqdm 是 Python 进度条库，可以在 Python 长循环中添加一个进度提示信息。
        bar = tqdm(enumerate(valid_dataloader), total=total_batch)
        for i, data in bar:
            img, gt = data['image'], data['label']

            if opt.use_gpu:
                img = img.cuda()
                gt = gt.cuda()

            output = model(img)
            _recall, _specificity, _precision, _F1, _F2, \
            _ACC_overall, _IoU_poly, _IoU_bg, _IoU_mean = evaluate(output, gt)

            metrics.update(recall= _recall, specificity= _specificity, precision= _precision, 
                            F1= _F1, F2= _F2, ACC_overall= _ACC_overall, IoU_poly= _IoU_poly, 
                            IoU_bg= _IoU_bg, IoU_mean= _IoU_mean
                        )

    metrics_result = metrics.mean(total_batch)

    return metrics_result


def train():

    model = generate_model(opt)

    # load data
    train_data = getattr(datasets, opt.dataset)(opt.root, opt.train_data_dir, mode='train')
    train_dataloader = DataLoader(train_data, opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    valid_data = getattr(datasets, opt.dataset)(opt.root, opt.valid_data_dir, mode='valid')
    valid_dataloader = DataLoader(valid_data, batch_size=1, shuffle=False, num_workers=opt.num_workers)
    val_total_batch = int(len(valid_data) / 1) # 常规除
   

    # load optimizer and scheduler
    optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.mt, weight_decay=opt.weight_decay)

    lr_lambda = lambda epoch: 1.0 - pow((epoch / opt.nEpoch), opt.power)
    scheduler = LambdaLR(optimizer, lr_lambda)

    # train
    print('Start training')
    print('---------------------------------\n')
    best_loss=1e5
    best_dice=0
    for epoch in range(opt.nEpoch):
        print('------ Epoch', epoch + 1)
        model.train()
        total_batch = int(len(train_data) / opt.batch_size)
        bar = tqdm(enumerate(train_dataloader), total=total_batch)
        epoch_loss=0
        step=0
        for i, data in bar:
            img = data['image']
            gt = data['label']
        

            if opt.use_gpu:
                img = img.cuda()
                gt = gt.cuda()

            optimizer.zero_grad()
            output = model(img)

            #loss = BceDiceLoss()(output, gt)
            loss = DeepSupervisionLoss(output, gt)
            epoch_loss = epoch_loss + loss.item()
            step+=1
            loss.backward()

            optimizer.step()
            bar.set_postfix_str('loss: %.5s' % loss.item())

        scheduler.step()

        metrics_result = valid(model, valid_dataloader, val_total_batch)
        

        print("Valid Result:")
        print('recall: %.4f, specificity: %.4f, precision: %.4f, F1: %.4f,'
              ' F2: %.4f, ACC_overall: %.4f, IoU_poly: %.4f, IoU_bg: %.4f, IoU_mean: %.4f'
              % (metrics_result['recall'], metrics_result['specificity'], metrics_result['precision'],
                 metrics_result['F1'], metrics_result['F2'], metrics_result['ACC_overall'],
                 metrics_result['IoU_poly'], metrics_result['IoU_bg'], metrics_result['IoU_mean']))
        # make checkpoints path
        save_path='./checkpoints/exp' + str(opt.expID)
        if not exists(save_path):
            os.makedirs(save_path)
            print(save_path,'  be established!')
        #save checkpoint of best loss
        if epoch_loss/step < best_loss:
            best_loss=epoch_loss/step
            print("Now best loss :",best_loss)
            torch.save(model.state_dict(), save_path+'/ck_bestloss.pth')
            print("Update bestloss checkpoint!",save_path+'/ck_bestloss.pth')
        #save checkpoint of best dice
        if best_dice < metrics_result['F1']:
            best_dice=metrics_result['F1']
            print("Now best dice is ",best_dice)
            torch.save(model.state_dict(),save_path+'/ck_bestdice.pth')
            print("Update bestloss checkpoint!",save_path+'/ck_bestdice.pth')
        if ((epoch + 1) % opt.ckpt_period == 0): 
            torch.save(model.state_dict(), './checkpoints/exp' + str(opt.expID)+"/ck_{}.pth".format(epoch + 1))
        #torch.cuda.empty_cache()
            


if __name__ == '__main__':

    if opt.mode == 'train':
        print('---PolpySeg Train---')
        train()

    print('Done')

