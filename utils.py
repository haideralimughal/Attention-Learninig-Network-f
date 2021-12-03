from torch.utils.data import DataLoader
import sys
import os
from torch.nn import BCELoss
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
import torch
from matplotlib import pyplot as plt
import imgaug.augmenters as iaa
from tqdm import tqdm
from .metrics import *
from .dataloader import TestNetDataset
from .model import TestNet
# from focal_loss.focal_loss import FocalLoss
from torch.utils.tensorboard import SummaryWriter

device = torch.device('cuda:0')

def train(args):
    # augmentations
    transforms = iaa.Sequential([
        iaa.Rotate((-15., 15.)),
        iaa.TranslateX(percent=(-0.05, 0.05)),
        iaa.TranslateY(percent=(-0.05, 0.05)),
        iaa.Affine(shear=(-50, 50)),
        iaa.Affine(scale=(0.8, 1.2)),
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5)
    ])

    # load data and create data loaders
    train_set = TestNetDataset(args.train_data, 'gland', batchsize=args.batch_size, steps=args.steps,
                              transforms=transforms)
    test_set = TestNetDataset(args.valid_data, args.valid_dataset)
    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=False, drop_last=True,
                              num_workers=0, pin_memory=True)
    test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=False, num_workers=0,
                             pin_memory=True)

    # create model
    model = TestNet(n_classes=1).to(device).float()

    criterion = BCELoss()
    #criterion = FocalLoss(alpha=2, gamma=5)
    optimizer = Adam(params=model.parameters(), lr=args.lr, weight_decay=0.)

    writer = SummaryWriter(f'/home/ali/Project_4/checkpoints/1')

    # keras lr decay equivalent
    fcn = lambda step: 1. / (1. + args.lr_decay * step)
    scheduler = LambdaLR(optimizer, lr_lambda=fcn)

    print('model successfully built and compiled.')

    if not os.path.isdir("checkpoints/" + args.exp):
        os.mkdir("checkpoints/" + args.exp)

    best_iou = 0.
    steps = 0

    print('\nStart training...')
    for epoch in range(args.epochs):
        tot_loss = 0.
        tot_iou = 0.
        tot_dice = 0.
        tot_jacc_v2 = 0.
        tot_f1 = 0.
        tot_pre = 0.
        tot_re = 0.
        tot_speci = 0.
        val_loss = 0.
        val_iou = 0.
        val_dice = 0.
        val_jacc_v2 = 0.
        val_f1 = 0.
        val_pre = 0.
        val_re = 0.
        val_speci = 0.


        # training
        model.train()
        for step, (x, y) in enumerate(
                tqdm(train_loader, desc='[TRAIN] Epoch ' + str(epoch + 1) + '/' + str(args.epochs))):
            if step >= args.steps:
                break
            x = x.to(device).float()
            y = y.to(device).float()

            optimizer.zero_grad()
            output = model(x)

            # loss
            l = criterion(output, y)
            tot_loss += l.item()
            l.backward()
            optimizer.step()

            # metrics
            x, y = output.detach().cpu().numpy(), y.detach().cpu().numpy()
            iou_score = iou(y, x)
            dice_score = dice_coef(y, x)
            jacc_score = Jaccard_coef_V2(y, x)
            f1_score = f1(y, x)
            pre_score = precision(y, x)
            re_score = recall(y, x)
            speci_score = Specificiy(y, x)
            tot_iou += iou_score
            tot_dice += dice_score
            tot_jacc_v2 += jacc_score
            tot_f1 += f1_score
            tot_pre += pre_score
            tot_re += re_score
            tot_speci += speci_score

            scheduler.step()

            writer.add_scalar('Training loss', l, global_step=steps)
            steps += 1

        print('[TRAIN] Epoch: ' + str(epoch + 1) + '/' + str(args.epochs),
              'loss:', tot_loss / args.steps,
              'iou:', tot_iou / args.steps,
              'dice:', tot_dice / args.steps,
              'Jacc:', tot_jacc_v2 / args.steps,
              'f1:', tot_f1 / args.steps,
              'Precision:', tot_pre / args.steps,
              'Recall:', tot_re / args.steps,
              'Specificity', tot_speci / args.steps)

        # validation
        model.eval()
        with torch.no_grad():
            for step, (x, y) in enumerate(
                    tqdm(test_loader, desc='[VAL] Epoch ' + str(epoch + 1) + '/' + str(args.epochs))):
                x = x.to(device).float()
                y = y.to(device).float()

                output = model(x)

                # loss
                l = criterion(output, y)
                val_loss += l.item()

                writer.add_scalar('validation loss', l, global_step=steps)


                # metrics
                x, y = output.detach().cpu().numpy(), y.cpu().numpy()
                iou_score = iou(y, x)
                dice_score = dice_coef(y, x)
                f1_score = f1(y, x)
                jacc_score = Jaccard_coef_V2(y, x)
                pre_score = precision(y, x)
                re_score = recall(y, x)
                speci_score = Specificiy(y, x)
                val_iou += iou_score
                val_dice += dice_score
                val_jacc_v2 += jacc_score
                val_f1 += f1_score
                val_pre += pre_score
                val_re += re_score
                val_speci += speci_score

        if val_iou / len(test_loader) > best_iou:
            best_iou = val_iou / len(test_loader)
            save_model(args, model)

        print('[VAL] Epoch: ' + str(epoch + 1) + '/' + str(args.epochs),
              'val_loss:', val_loss / len(test_loader),
              'val_iou:', val_iou / len(test_loader),
              'val_dice:', val_dice / len(test_loader),
              'va_jacc:', val_jacc_v2 / len(test_loader),
              'val_f1:', val_f1 / len(test_loader),
              'val_pre:', val_pre / len(test_loader),
              'val_re:', val_re / len(test_loader),
              'val_speci', val_speci / len(test_loader),
              'best val_iou:', best_iou)

    print('\nTraining fininshed!')


def evaluate(args):
    # load data and create data loader
    test_set = TestNetDataset(args.valid_data, args.valid_dataset)
    test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=False, num_workers=0,
                             pin_memory=True)

    if args.model_path is None:
        integrate = '_int' if args.integrate else ''
        weights = '_weights'
        cpt_name = 'iter_' + str(args.iter) + '_mul_' + str(args.multiplier) + integrate + '_best' + weights + '.pt'
        model_path = "checkpoints/" + args.exp + "/" + cpt_name
    else:
        model_path = args.model_path
    print('Restoring model from path: ' + model_path)
    model = TestNet(n_classes=1).to(device)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])

    criterion = BCELoss()

    val_loss = 0.
    val_iou = 0.
    val_dice = 0.
    val_jacc_v2 = 0.
    val_f1 = 0.
    val_pre = 0.
    val_re = 0.
    val_speci = 0.

    segmentations = []

    writer = SummaryWriter(f'/home/ali/Project_4/checkpoints/1')
    steps = 0

    # validation
    print('\nStart evaluation...')
    model.eval()
    with torch.no_grad():
        for step, (x, y) in enumerate(tqdm(test_loader)):

            x = x.to(device).float()
            y = y.to(device).float()

            output = model(x)

            # loss
            l = criterion(output, y)
            val_loss += l.item()

            writer.add_scalar('validation loss', l, global_step=steps)
            steps +=1

            # metrics
            x, y = output.detach().cpu().numpy(), y.cpu().numpy()
            iou_score = iou(y, x)
            dice_score = dice_coef(y, x)
            jacc_v2_score = Jaccard_coef_V2(y, x)
            f1_score = f1(y, x)
            pre_score = precision(y, x)
            re_score = recall(y, x)
            speci_score = Specificiy(y, x)
            val_iou += iou_score
            val_dice += dice_score
            val_jacc_v2 += jacc_v2_score
            val_f1 += f1_score
            val_pre += pre_score
            val_re += re_score
            val_speci += speci_score

            if args.save_result:
                segmentations.append(x)

    val_loss = val_loss / len(test_loader)
    val_iou = val_iou / len(test_loader)
    val_dice = val_dice / len(test_loader)
    val_jacc_v2 = val_jacc_v2 / len(test_loader)
    val_f1 = val_f1 / len(test_loader)
    val_pre = val_pre / len(test_loader)
    val_re = val_re / len(test_loader)
    val_speci = val_speci / len(test_loader)
    print('Validation loss:\t', val_loss)
    print('Validation  iou:\t', val_iou)
    print('Validation dice:\t', val_dice)
    print('Validation Jacc:\t', val_jacc_v2)
    print('Validation f1:\t', val_f1)
    print('Validation Precision:\t', val_pre)
    print('Validation Recall:\t', val_re)
    print('Validation Specificity:\t', val_speci)

    print('\nEvaluation finished!')

    if args.save_result:

        # save metrics
        if not os.path.exists("checkpoints/" + args.exp + "/outputs"):
            os.mkdir("checkpoints/" + args.exp + "/outputs")

        with open("checkpoints/" + args.exp + "/outputs/result.txt", 'w+') as f:
            f.write('Validation loss:\t' + str(val_loss) + '\n')
            f.write('Validation  iou:\t' + str(val_iou) + '\n')
            f.write('Validation dice:\t' + str(val_dice) + '\n')
            f.write('Validation Jacc:\t' + str(val_jacc_v2) + '\n')
            f.write('Validation f1:\t' + str(val_f1) + '\n')
            f.write('Validation Precision:\t' + str(val_pre) + '\n')
            f.write('Validation Recall:\t' + str(val_re) + '\n')
            f.write('Validation Specificity:\t' + str(val_speci) + '\n')

        print('Metrics have been saved to:', "checkpoints/" + args.exp + "/outputs/result.txt")

        # save segmentations
        results = np.transpose(np.concatenate(segmentations, axis=0), (0, 2, 3, 1))
        results = (results > 0.5).astype(np.float32)  # Binarization. Comment out this line if you don't want to

        print('Saving segmentations...')
        if not os.path.exists("checkpoints/" + args.exp + "/outputs/segmentations"):
            os.mkdir("checkpoints/" + args.exp + "/outputs/segmentations")

        for i in range(results.shape[0]):
            plt.imsave("checkpoints/" + args.exp + "/outputs/segmentations/" + str(i) + ".png", results[i, :, :, 0],
                       cmap='gray')  # binary segmenation

        print('A total of ' + str(results.shape[0]) + ' segmentation results have been saved to:',
              "checkpoints/" + args.exp + "/outputs/segmentations/")


def save_model(args, model):
    integrate = '_int' if args.integrate else ''
    weights = '_weights'
    cpt_name = 'iter_' + str(args.iter) + '_mul_' + str(args.multiplier) + integrate + '_best' + weights + '.pt'
    torch.save({'state_dict': model.state_dict()}, "checkpoints/" + args.exp + "/" + cpt_name)
