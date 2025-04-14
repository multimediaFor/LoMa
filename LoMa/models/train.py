import numpy as np
from mydatasets import MyDataset
from torch.utils.data import DataLoader
import argparse
import logging as logger
import torch.optim as optim
import torch.nn as nn
import torch
import os
import shutil
from loss import MyLoss
from vmamba import MyModel
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from metric import calc_fixed_f1_iou
import datetime

logger.basicConfig(level=logger.INFO,
                   format='%(levelname)s %(asctime)s] %(message)s',
                   datefmt='%m-%d %H:%M:%S')

parser = argparse.ArgumentParser()
parser.add_argument('--input_size', type=int, default=512, help='size of resized input')
parser.add_argument('--gt_ratio', type=int, default=1, help='resolution of input / output')
parser.add_argument('--train_bs', type=int, default=16, help='training batch size')
parser.add_argument('--test_bs', type=int, default=16, help='testing batch size')
parser.add_argument('--flist_path', type=str, default='./flist/', help='data set path')
parser.add_argument('--gpu', type=str, default='0', help='GPU ID')

args = parser.parse_args()
logger.info(args)

date_now = datetime.datetime.now()
date_now = 'Log_v%02d%02d%02d%02d/' % (date_now.month, date_now.day, date_now.hour, date_now.minute)
args.out_dir = date_now

device = torch.device('cuda:{}'.format(args.gpu))

np.random.seed(666666)
torch.manual_seed(666666)
torch.cuda.manual_seed(666666)
torch.backends.cudnn.deterministic = True


def rm_and_make_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


class MyVmamba(nn.Module):
    def __init__(self, net_weight="/data/myfile/LoMa/classification/models/weights/efficient_vmamba_base.ckpt"):
        super(MyVmamba, self).__init__()
        self.lr = 1e-4
        self.cur_net = MyModel().to(device)
        if net_weight is not None:
            self.load(self.cur_net, net_weight)
        self.optimizer = optim.AdamW(self.cur_net.parameters(), lr=self.lr)
        self.save_dir = 'weights/' + args.out_dir
        rm_and_make_dir(self.save_dir)

    def process(self, Ii, Mg=None, isTrain=False):
        self.optimizer.zero_grad()
        if isTrain:
            Fo = self.cur_net(Ii)
            batch_loss = MyLoss(Fo, Mg)
            self.backward(batch_loss)
            return batch_loss
        else:
            with torch.no_grad():
                Fo = self.cur_net(Ii)
            return Fo

    def backward(self, batch_loss=None):
        if batch_loss:
            batch_loss.backward(retain_graph=False)
            self.optimizer.step()

    def save(self, path):
        if not os.path.exists(self.save_dir + path):
            os.makedirs(self.save_dir + path)
        torch.save(self.cur_net.state_dict(),
                   self.save_dir + path + 'weights.pth')

    def load(self, model, path):
        weights = torch.load(path)['state_dict']
        model_state_dict = model.state_dict()
        loaded_layers = []
        missing_layers = []
        mismatched_shapes = []

        # 遍历加载的权重字典
        for name, param in weights.items():
            name = 'encoder.' + name
            if name in model_state_dict:
                if param.shape == model_state_dict[name].shape:
                    model_state_dict[name].copy_(param)  # 更新模型的权重
                    loaded_layers.append(name)
                else:
                    mismatched_shapes.append(name)
            else:
                # 如果模型中没有该层，记录缺失的层
                missing_layers.append(name)

        # 打印加载成功的层
        if loaded_layers:
            logger.info(f"Successfully loaded the following layers: {', '.join(loaded_layers)}")

        # 打印形状不匹配的层
        if mismatched_shapes:
            logger.warning(f"The following layers have mismatched shapes: {', '.join(mismatched_shapes)}")

        # 打印缺失的层
        if missing_layers:
            logger.warning(f"The following layers are missing in the model: {', '.join(missing_layers)}")

        # 如果都加载成功，打印成功信息
        if not mismatched_shapes and not missing_layers:
            logger.info("All layers have been successfully loaded!")


class ForgeryForensics():
    def __init__(self):
        self.LoMa = MyVmamba().to(device)
        self.n_epochs = 99999
        self.train_npy_list = [
            # name, repeat_time
            ("sp_images_199999.npy", 1),
            ("cm_images_199429.npy", 1),
            ("bcm_images_199443.npy", 1),
            ("CASIA2_5123.npy", 40),
            ('IMD20_2010.npy', 20),
        ]
        self.train_file = None
        for item in self.train_npy_list:
            self.train_file_tmp = np.load(args.flist_path + item[0])
            for _ in range(item[1]):
                self.train_file = np.concatenate(
                    [self.train_file, self.train_file_tmp]) if self.train_file is not None else self.train_file_tmp
        self.train_num = len(self.train_file)
        train_dataset = MyDataset(num=self.train_num, file=self.train_file, choice='train',
                                  input_size=args.input_size, gt_ratio=args.gt_ratio)

        self.val_npy_list = [
            # name, nickname
            
            ('DSO_100.npy', 'DSO'),
            ('CASIA1_920.npy', 'CASIAv1'),
            ('FFpp_1000.npy', 'FFpp'),
        ]
        self.val_file_list = []
        for item in self.val_npy_list:
            self.val_file_tmp = np.load(args.flist_path + item[0])
            self.val_file_list.append(self.val_file_tmp)

        self.train_bs = args.train_bs
        self.test_bs = args.test_bs

        self.train_loader = DataLoader(dataset=train_dataset, batch_size=self.train_bs, shuffle=True)
        logger.info('Train on %d images.' % self.train_num)
        for idx, file_list in enumerate(self.val_file_list):
            logger.info('Test on %s (#%d).' % (self.val_npy_list[idx][0], len(file_list)))

    def train(self):
        train_writer = SummaryWriter(log_dir=os.path.join(self.LoMa.save_dir, 'runs'))
        count, batch_losses = 0, []  # count是训练集样本个数
        best_score = 0
        scheduler = ReduceLROnPlateau(self.LoMa.optimizer, mode='max', factor=0.8, patience=3,
                                      min_lr=1e-8)
        self.LoMa.train()
        
        for epoch in range(1, self.n_epochs + 1):
            for items in self.train_loader:
                count += self.train_bs
                Ii, Mg = (item.to(device) for item in items[:2])  # Input, Ground-truth Mask

                batch_loss = self.LoMa.process(Ii, Mg, isTrain=True)
                batch_losses.append(batch_loss.item())  # 存储部分数据集的loss，不是全部数据集
                if count % (self.train_bs * 5) == 0:
                    # 记录学习率
                    current_lr = self.LoMa.optimizer.param_groups[0]['lr']
                    logger.info('Train Num (%6d/%6d), Loss:%5.4f, LR: %5.8f' % (
                        count, self.train_num, np.mean(batch_losses), current_lr))
                    train_writer.add_scalar('Loss/train', np.mean(batch_losses), count, current_lr)

                if count % int((self.train_loader.dataset.__len__() / 100) // self.train_bs * self.train_bs) == 0:
                    
                    self.LoMa.save('latest/')
                    logger.info(
                        'Ep%03d(%6d/%6d): Tra: Loss :%5.4f' % (epoch, count, self.train_num, np.mean(batch_losses)))
                    train_writer.add_scalar('Loss/train', np.mean(batch_losses), count)
                    tmp_score = self.val(epoch)
                    scheduler.step(tmp_score)
                    if tmp_score > best_score:
                        train_writer.add_scalar('Score/train', tmp_score, count)
                        best_score = tmp_score
                        logger.info('Score: %5.4f (Best)' % best_score)
                        train_writer.add_scalar('Score/train(Best)', best_score, count)
                        self.LoMa.save('Ep%03d_%5.4f/' % (epoch, tmp_score))
                    else:
                        logger.info('Score: %5.4f' % tmp_score)
                        train_writer.add_scalar('Score/train', tmp_score, count)
                    self.LoMa.train()
                    batch_losses = []
            count = 0

    def val(self, epoch):
        tmp_F1 = []
        tmp_IOU = []
        test_nums = []
        result_file_path = os.path.join(self.LoMa.save_dir, 'result.txt')
        with open(result_file_path, 'a') as result_file:  # Open the file for appending
            result_file.write(f"Epoch {epoch}:\n")
            for idx in range(len(self.val_file_list)):
                P_F1, P_IOU, test_num = ForensicTesting(self.LoMa, bs=self.test_bs,
                                                        test_npy=self.val_npy_list[idx][0],
                                                        test_file=self.val_file_list[idx])
                tmp_IOU.append(P_IOU)
                tmp_F1.append(P_F1)
                test_nums.append(test_num)
                result_str = '%s(#%d): F1:%5.4f, IoU:%5.4f\n' % (
                    self.val_npy_list[idx][1],
                    len(self.val_file_list[idx]),
                    P_F1,  # F1 score
                    P_IOU  # IoU score
                )
                result_file.write(result_str)

            # 加权平均计算
            total_samples = sum(test_nums)  # 所有测试集的样本总数
            weighted_avg_F1 = np.sum(np.array(tmp_F1) * np.array(test_nums)) / total_samples
            weighted_avg_IOU = np.sum(np.array(tmp_IOU) * np.array(test_nums)) / total_samples
            logger.info('Average weight F1: %5.4f' % weighted_avg_F1)
            logger.info('Average weight IoU: %5.4f' % weighted_avg_IOU)
            avg_weight_result_str = 'Average weight F1: %5.4f\nAverage weight IoU: %5.4f\n\n' % (
                weighted_avg_F1, weighted_avg_IOU)
            result_file.write(avg_weight_result_str)

        return (weighted_avg_F1 + weighted_avg_IOU) / 2.0


# test
def ForensicTesting(model, bs=1, test_npy='', test_file=None):
    if test_file is None:
        test_file = np.load(args.flist_path + test_npy)
    test_num = len(test_file)
    test_dataset = MyDataset(test_num, test_file, choice='test')
    test_loader = DataLoader(dataset=test_dataset, batch_size=bs, num_workers=min(48, 16), shuffle=False)
    model.eval()

    f1, iou = [], []

    for items in test_loader:
        Ii, Mg, Hg, Wg = (item.to(device) for item in items[:-1])
        with torch.no_grad():
            Mo = model.process(Ii)
        for i in range(Mo.shape[0]):
            fixed_f1, iou_score = calc_fixed_f1_iou(Mo[i], Mg[i])
            f1.append(fixed_f1.cpu())
            iou.append(iou_score.cpu())

    Pixel_F1 = np.mean(f1)
    Pixel_IOU = np.mean(iou)
    return Pixel_F1, Pixel_IOU, test_num


if __name__ == '__main__':
    model = ForgeryForensics()
    model.train()
