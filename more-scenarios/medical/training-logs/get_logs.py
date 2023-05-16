from torch.utils.tensorboard import SummaryWriter

logdir = 'D:/Cyz/Github/UniMatch-main/UniMatch-main/more-scenarios/medical/training-logs'  # TensorBoard的日志目录
logfile = 'D:/Cyz/Github/UniMatch-main/UniMatch-main/more-scenarios/medical/training-logs/1case.log'   # .log文件路径

# 创建一个写入器（writer）对象
writer = SummaryWriter(log_dir=logdir)

# 读取.log文件
with open(logfile, 'r') as f:
    lines = f.readlines()

# 遍历.log文件每一行
for line in lines:

    # 如果包含Epoch关键字，则提取epoch信息
    if 'Epoch' in line:
        epoch = int(line.split(':')[0].split()[-1])

    # 如果包含Evaluation关键字，则提取评估指标信息
    # ***** Evaluation ***** >>>> Class [0 Right Ventricle] Dice: 59.08
    # ***** Evaluation ***** >>>> Class [1 Myocardium] Dice: 66.93
    # ***** Evaluation ***** >>>> Class [2 Left Ventricle] Dice: 79.71
    # ***** Evaluation ***** >>>> MeanDice: 68.57
    # elif 'Evaluation' in line:
    #     # dices = [float(dice.split(':')[-1]) for dice in line.split('>>>>') if 'Dice' in dice]
    #     # mean_dice = sum(dices) / len(dices)
    #
    #     # 定义一个字典，用于存储各个类别的 dice 值
    #     dice_dict = []
    #     # 根据字符串中的回车符进行分段，得到一个列表
    #     segments = line.split("\n")
    #
    #     # 遍历列表中的每个元素（即每行字符串）
    #     for seg in segments:
    #         # 如果该行字符串包含关键字 Class，说明该行包含某个类别的 dice 值
    #         if "Class" in seg:
    #             # 对字符串进行分割和截取，提取出类别名称和 dice 值
    #             class_name = seg.split("[")[1].split("]")[0]  # 获取类别名称
    #
    #             dice_value = float(seg.strip().split(":")[-1])  # 获取 dice 值
    #
    #             # 将类别名称和 dice 值存入字典中
    #             dice_dict[class_name] = dice_value
    #
    #         # 如果该行字符串包含关键字 MeanDice，说明该行包含整体的 MeanDice 值
    #         elif "MeanDice" in seg:
    #             mean_dice = float(seg.strip().split(":")[-1])
    #
    #             # 输出各个类别的 dice 值和整体的 MeanDice 值
    #             print("Right Ventricle Dice:", dice_dict[0])
    #             print("Myocardium Dice:", dice_dict[1])
    #             print("Left Ventricle Dice:", dice_dict[2])
    #             print("MeanDice:", mean_dice)
    #
    #     # 将评估指标写入summary
    #
    #     writer.add_scalar('right_ventricle_dice', dice_dict[0], epoch)
    #     writer.add_scalar('myocardium_dice', dice_dict[1], epoch)
    #     writer.add_scalar('left_ventricle_dice', dice_dict[2], epoch)
    #     writer.add_scalar('mean_dice', mean_dice, epoch)

    # 如果包含Iters关键字，则提取损失和精度信息
    # Iters: 0, Total loss: 0.034, Loss x: 0.033, Loss s: 0.061, Loss w_fp: 0.009, Mask ratio: 0.971
    elif 'Iters' in line:
        iters = int(line.split(':')[0].split()[-1])
        total_loss = float(line.split(':')[1].split(',')[0].strip().split()[-1])
        loss_x = float(line.split(':')[2].split(',')[0].strip().split()[-1])
        # loss_s = float(line.split(':')[1].split(',')[0].strip().split()[-1])
        loss_w_fp = float(line.split(':')[4].split(',')[0].strip().split()[-1])
        mask_ratio = float(line.split(':')[-1].strip())


        # 将损失和精度信息写入summary
        writer.add_scalar('train/total_loss', total_loss, iters)
        writer.add_scalar('train/loss_x', loss_x, iters)
        # writer.add_scalar('train/loss_s', (loss_u_s1 + loss_u_s2)) / 2.0, iters)
        writer.add_scalar('train/loss_w_fp', loss_w_fp, iters)
        writer.add_scalar('mask_ratio', mask_ratio, iters)


