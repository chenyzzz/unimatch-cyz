import argparse
import logging
import os
import pprint

import torch
from torch import nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml

from dataset.acdc import ACDCDataset
from model.unet import UNet
from util.classes import CLASSES
from util.utils import AverageMeter, count_params, init_log, DiceLoss, TI_Loss
from util.dist_helper import setup_distributed

parser = argparse.ArgumentParser(
    description='Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation')
parser.add_argument('--config',
                    default='D:/Cyz/Github/UniMatch-main/UniMatch-main/more-scenarios/medical/configs/acdc.yaml',
                    type=str, required=False)
parser.add_argument('--labeled-id-path', default='splits/acdc/1/labeled.txt', type=str, required=False)
parser.add_argument('--unlabeled-id-path', default='splits/acdc/1/unlabeled.txt', type=str, required=False)
parser.add_argument('--save-path', default='exp/acdc-save/unimatch/unet/split-1', type=str, required=False)
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--port', default=20024, type=int)

parser.add_argument('--use_TI_loss', action='store_true', help="Use TI loss or not")
parser.add_argument('--TI_weight', default=1e-6, help="TI loss weight")

parser.add_argument('--log-file', default='exp/acdc-save/unimatch/unet/split-1/1.log', type=str, required=False,
                    help='Path to the log file')


# parser.add_argument('--nproc_per_node', default=1, type=int)
# parser.add_argument('--master_addr', default='localhost', type=str)
# parser.add_argument('--master_port', default=20024, type=int)


def main():
    args = parser.parse_args()

    cfg = yaml.load(open(args.config, "r", encoding='utf-8'), Loader=yaml.Loader)

    logger = init_log('global', logging.INFO, args.log_file)
    logger.propagate = 0  # 该日志记录器不会将消息传递给父处理程序

    # 调用 setup_distributed 函数，设置分布式训练环境，获取当前进程号 rank 和总的进程数 world_size。
    # 其中 port 为端口号，用于进程间通信。
    # rank, world_size = setup_distributed(port=args.port)

    # 将 rank 和 world_size 设为 0 和 1，即不启用分布式训练
    rank, world_size = 0, 1

    if rank == 0:
        all_args = {**cfg, **vars(args), 'ngpus': world_size}
        logger.info('{}\n'.format(pprint.pformat(all_args)))

        writer = SummaryWriter(args.save_path)

        os.makedirs(args.save_path, exist_ok=True)

    cudnn.enabled = True
    cudnn.benchmark = True  # 设置 PyTorch 使用 CuDNN 加速，可以加速一些基于卷积神经网络的计算

    model = UNet(in_chns=1, class_num=cfg['nclass'])
    if rank == 0:
        logger.info('Total params: {:.1f}M\n'.format(count_params(model)))

    optimizer = SGD(model.parameters(), cfg['lr'], momentum=0.9, weight_decay=0.0001)

    # 将 local_rank 设为 0，即使用第一个 GPU
    # local_rank = int(os.environ["LOCAL_RANK"])
    local_rank = 0

    # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()  # 模型移动到GPU上

    # 不使用 DistributedDataParallel 包装模型
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False,
    #                                                   output_device=local_rank, find_unused_parameters=False)

    criterion_ce = nn.CrossEntropyLoss()
    criterion_dice = DiceLoss(n_classes=cfg['nclass'])

    if args.use_TI_loss:
        criterion_ti = TI_Loss(dim=2, connectivity=4, inclusion=[[2, 3]], exclusion=[[1, 3], [1, 2]])
    else:
        criterion_ti = None

    trainset_u = ACDCDataset(cfg['dataset'], cfg['data_root'], 'train_u',
                             cfg['crop_size'], args.unlabeled_id_path)  # unlabel 无标签
    trainset_l = ACDCDataset(cfg['dataset'], cfg['data_root'], 'train_l',
                             cfg['crop_size'], args.labeled_id_path, nsample=len(trainset_u.ids))  # 有标签
    valset = ACDCDataset(cfg['dataset'], cfg['data_root'], 'val')

    # trainsampler_l = torch.utils.data.distributed.DistributedSampler(trainset_l)

    trainloader_l = DataLoader(trainset_l, batch_size=cfg['batch_size'],
                               pin_memory=True, num_workers=1, drop_last=True)  # sampler=trainsampler_l)
    # trainsampler_u = torch.utils.data.distributed.DistributedSampler(trainset_u)
    trainloader_u = DataLoader(trainset_u, batch_size=cfg['batch_size'],
                               pin_memory=True, num_workers=1, drop_last=True)  # sampler=trainsampler_u)
    # trainsampler_u_mix = torch.utils.data.distributed.DistributedSampler(trainset_u)
    trainloader_u_mix = DataLoader(trainset_u, batch_size=cfg['batch_size'],
                                   pin_memory=True, num_workers=1,
                                   drop_last=True)  # sampler=trainsampler_u_mix)  有标注和无标注混合的数据集加载器
    # valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=1,
                           drop_last=False)  # sampler=valsampler)

    total_iters = len(trainloader_u) * cfg[
        'epochs']  # 总迭代次数=无标注数据集迭代器中每个样本都被训练一遍所需要的迭代次数，等于无标注数据集长度 len(trainloader_u) 乘以总共训练的 epochs 数量 cfg['epochs']
    previous_best = 0.0
    epoch = -1  # 将当前 epochs 数量 epoch 设为 -1，表示还没有开始训练。

    if os.path.exists(os.path.join(args.save_path, 'latest.pth')):
        checkpoint = torch.load(os.path.join(args.save_path, 'latest.pth'))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        previous_best = checkpoint['previous_best']
        # 如果存在已保存的checkpoint，则加载checkpoint恢复模型及其他参数
        if rank == 0:
            logger.info('************ Load from checkpoint at epoch %i\n' % epoch)

    for epoch in range(epoch + 1, cfg['epochs']):  # 循环训练直至达到指定的epochs数量
        if rank == 0:
            logger.info('===========> Epoch: {:}, LR: {:.5f}, Previous best: {:.2f}'.format(
                epoch, optimizer.param_groups[0]['lr'], previous_best))

        total_loss = AverageMeter()
        total_loss_x = AverageMeter()
        total_loss_s = AverageMeter()
        total_loss_w_fp = AverageMeter()
        total_mask_ratio = AverageMeter()  # 无标签数据中高置信度像素点的比例

        # 打乱数据集并设置随机种子，确保每个epoch中的数据加载顺序是不同的
        # trainloader_l.sampler.set_epoch(epoch)
        # trainloader_u.sampler.set_epoch(epoch)
        # trainloader_u_mix.sampler.set_epoch(epoch + cfg['epochs'])
        # 打包成一个迭代器
        loader = zip(trainloader_l, trainloader_u, trainloader_u_mix)

        for i, ((img_x, mask_x),  # 有标注数据集
                (img_u_w, img_u_s1, img_u_s2, cutmix_box1, cutmix_box2),  # 无标注训练集
                (img_u_w_mix, img_u_s1_mix, img_u_s2_mix, _, _)) in enumerate(loader):  # 进行MixMatch训练时随机生成的无标注训练集

            img_x, mask_x = img_x.cuda(), mask_x.cuda()
            img_u_w = img_u_w.cuda()
            img_u_s1, img_u_s2 = img_u_s1.cuda(), img_u_s2.cuda()
            cutmix_box1, cutmix_box2 = cutmix_box1.cuda(), cutmix_box2.cuda()
            img_u_w_mix = img_u_w_mix.cuda()
            img_u_s1_mix, img_u_s2_mix = img_u_s1_mix.cuda(), img_u_s2_mix.cuda()

            with torch.no_grad():
                model.eval()
                # 使用无梯度计算的no_grad()上下文管理器和model.eval()方法进入模型评估模式，
                # 以避免Batch Normalization层中的均值和方差被随着训练而更新导致结果不稳定。

                pred_u_w_mix = model(img_u_w_mix).detach()  # 生成的无标注训练集img_u_w_mix输入到深度神经网络模型中，得到pred_u_w_mix预测矩阵
                conf_u_w_mix = pred_u_w_mix.softmax(dim=1).max(dim=1)[0]  # 计算每个像素点的置信度
                mask_u_w_mix = pred_u_w_mix.argmax(dim=1)  # 计算每个像素点的预测值

            # 使用MixMatch方法对无标注数据集进行增强
            img_u_s1[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1] = \
                img_u_s1_mix[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1]
            img_u_s2[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1] = \
                img_u_s2_mix[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1]

            model.train()

            num_lb, num_ulb = img_x.shape[0], img_u_w.shape[0]

            # 将有标注和无标注数据集合并输入深度神经网络模型中，得到预测矩阵。
            preds, preds_fp = model(torch.cat((img_x, img_u_w)), True)
            pred_x, pred_u_w = preds.split([num_lb, num_ulb])  # 分成有标注和无标注部分
            pred_u_w_fp = preds_fp[num_lb:]  # 得到了在无标注部分上的预测结果的分支

            pred_u_s1, pred_u_s2 = model(torch.cat((img_u_s1, img_u_s2))).chunk(2)  # 分成有标注和无标注部分

            pred_u_w = pred_u_w.detach()
            conf_u_w = pred_u_w.softmax(dim=1).max(dim=1)[0]
            mask_u_w = pred_u_w.argmax(dim=1)

            mask_u_w_cutmixed1, conf_u_w_cutmixed1 = mask_u_w.clone(), conf_u_w.clone()
            mask_u_w_cutmixed2, conf_u_w_cutmixed2 = mask_u_w.clone(), conf_u_w.clone()

            mask_u_w_cutmixed1[cutmix_box1 == 1] = mask_u_w_mix[cutmix_box1 == 1]
            conf_u_w_cutmixed1[cutmix_box1 == 1] = conf_u_w_mix[cutmix_box1 == 1]

            mask_u_w_cutmixed2[cutmix_box2 == 1] = mask_u_w_mix[cutmix_box2 == 1]
            conf_u_w_cutmixed2[cutmix_box2 == 1] = conf_u_w_mix[cutmix_box2 == 1]

            # 有标注部分使用交叉熵和Dice Loss的平均值
            loss_x = (criterion_ce(pred_x, mask_x)
                      + criterion_dice(pred_x.softmax(dim=1), mask_x.unsqueeze(1).float())
                      + criterion_ti(pred_x, mask_x.unsqueeze(1)) * args.TI_weight if criterion_ti else torch.tensor(0)
                      ) / (3.0 if criterion_ti else 2.0)

            # 无标注部分采用Dice Loss的平均值.利用阈值cfg['conf_thresh']过滤掉低置信度像素点
            # ignore=(conf_u_w_cutmixed1 < cfg['conf_thresh']).float()
            mask_u_w_cutmixed1[conf_u_w_cutmixed1 < cfg['conf_thresh']] = 0
            loss_u_s1 = criterion_dice(pred_u_s1.softmax(dim=1), mask_u_w_cutmixed1.unsqueeze(1).float()) \
                        + criterion_ti(pred_u_s1.softmax(dim=1), mask_u_w_cutmixed1.unsqueeze(1)) * args.TI_weight
            loss_u_s1 /= 2.0 if args.use_TI_loss else 1.0

            mask_u_w_cutmixed2[conf_u_w_cutmixed2 < cfg['conf_thresh']] = 0
            loss_u_s2 = criterion_dice(pred_u_s2.softmax(dim=1), mask_u_w_cutmixed2.unsqueeze(1).float()) \
                        + criterion_ti(pred_u_s2.softmax(dim=1), mask_u_w_cutmixed2.unsqueeze(1)) * args.TI_weight
            loss_u_s2 /= 2.0 if args.use_TI_loss else 1.0

            mask_u_w[conf_u_w < cfg['conf_thresh']] = 0
            loss_u_w_fp = criterion_dice(pred_u_w_fp.softmax(dim=1), mask_u_w.unsqueeze(1).float()) \
                          + criterion_ti(pred_u_w_fp.softmax(dim=1), mask_u_w.unsqueeze(1)) * args.TI_weight
            loss_u_w_fp /= 2.0 if args.use_TI_loss else 1.0

            loss = (loss_x + loss_u_s1 * 0.25 + loss_u_s2 * 0.25 + loss_u_w_fp * 0.5) / 2.0  # 计算总体损失函数loss,除以2.0来归一化

            # torch.distributed.barrier()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 更新损失函数的值
            total_loss.update(loss.item())
            total_loss_x.update(loss_x.item())
            total_loss_s.update((loss_u_s1.item() + loss_u_s2.item()) / 2.0)
            total_loss_w_fp.update(loss_u_w_fp.item())

            mask_ratio = (conf_u_w >= cfg['conf_thresh']).sum() / conf_u_w.numel()
            total_mask_ratio.update(mask_ratio.item())

            # 对学习率进行动态调整:在当前轮次完成后计算当前的迭代次数并根据总的迭代次数计算相应的学习率衰减系数
            iters = epoch * len(trainloader_u) + i
            lr = cfg['lr'] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr

            if rank == 0:
                writer.add_scalar('train/loss_all', loss.item(), iters)
                writer.add_scalar('train/loss_x', loss_x.item(), iters)  # 有标注数据集上的损失函数
                writer.add_scalar('train/loss_s', (loss_u_s1.item() + loss_u_s2.item()) / 2.0, iters)  # 无标注数据集上的损失函数
                writer.add_scalar('train/loss_w_fp', loss_u_w_fp.item(), iters)  # 特征增强部分的损失函数
                writer.add_scalar('train/mask_ratio', mask_ratio, iters)  # 满足一定置信度阈值的像素点占比等指标

            if (i % (len(trainloader_u) // 8) == 0) and (rank == 0):  # 每进行8次处理无标注数据的batch时输出当前的损失值结果
                logger.info(
                    'Iters: {:}, Total loss: {:.3f}, Loss x: {:.3f}, Loss s: {:.3f}, Loss w_fp: {:.3f}, Mask ratio: '
                    '{:.3f}'.format(i, total_loss.avg, total_loss_x.avg, total_loss_s.avg,
                                    total_loss_w_fp.avg, total_mask_ratio.avg))

        model.eval()  # 模型评估阶段
        dice_class = [0] * 3  # 三元素列表,分别表示三个类别（分类问题中的类别数目）的Dice系数之和

        with torch.no_grad():  # 禁止autograd计算梯度，可以节省内存并提高速度
            # 在每个验证图像上进行推理（不调整任何参数），并计算该图像的Dice系数。
            for img, mask in valloader:
                img, mask = img.cuda(), mask.cuda()

                h, w = img.shape[-2:]
                img = F.interpolate(img, (cfg['crop_size'], cfg['crop_size']), mode='bilinear', align_corners=False)

                img = img.permute(1, 0, 2, 3)

                pred = model(img)

                pred = F.interpolate(pred, (h, w), mode='bilinear', align_corners=False)
                pred = pred.argmax(dim=1).unsqueeze(0)

                for cls in range(1, cfg['nclass']):
                    # 使用inter和union计算分子和分母,预测结果和真实值的交集和并集
                    inter = ((pred == cls) * (mask == cls)).sum().item()
                    union = (pred == cls).sum().item() + (mask == cls).sum().item()
                    dice_class[cls - 1] += 2.0 * inter / union

        dice_class = [dice * 100.0 / len(valloader) for dice in
                      dice_class]  # dice_class是一个三元素列表，分别表示三个类别（分类问题中的类别数目）的Dice系数之和
        mean_dice = sum(dice_class) / len(dice_class)  # 是所有类别Dice系数的平均值

        if rank == 0:
            for (cls_idx, dice) in enumerate(dice_class):  # 输出三个类别的dice值
                logger.info('***** Evaluation ***** >>>> Class [{:} {:}] Dice: '
                            '{:.2f}'.format(cls_idx, CLASSES[cfg['dataset']][cls_idx], dice))
            logger.info('***** Evaluation ***** >>>> MeanDice: {:.2f}\n'.format(mean_dice))

            writer.add_scalar('eval/MeanDice', mean_dice, epoch)
            for i, dice in enumerate(dice_class):
                writer.add_scalar('eval/%s_dice' % (CLASSES[cfg['dataset']][i]), dice, epoch)

        is_best = mean_dice > previous_best
        previous_best = max(mean_dice, previous_best)
        if rank == 0:
            checkpoint = {  # 将当前模型的状态字典、优化器状态字典、当前轮次以及历史最佳平均Dice记录为字典checkpoint
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'previous_best': previous_best,
            }
            torch.save(checkpoint, os.path.join(args.save_path, 'latest.pth'))
            if is_best:
                torch.save(checkpoint, os.path.join(args.save_path, 'best.pth'))


if __name__ == '__main__':
    main()
