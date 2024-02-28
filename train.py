import argparse
import sys
import time
import numpy as np
from data.metrics import psnr,ssim
import warnings
import torch
import torch.backends.cudnn as cudnn
import torchvision.utils
from torch import optim, nn
from torch.utils.data import DataLoader
import utils
from data.dataloader import TrainDataloader_SR,TestDataloader
from networks import SRNetPlus,VGG19PerceptualLoss
from PIL import Image
from PIL import ImageFile
from torchvision import transforms
from torchvision.utils import save_image

cudnn.benchmark = True
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Disable OSError: image file is truncated
warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--clear_dir', type=str, default='./datasets/outdoor/train/clear/',help='!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
# parser.add_argument('--hazy_dir', type=str, default='./results/stylized_BeDDE/stylized200/',help='!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
parser.add_argument('--hazy_dir', type=str, default='./datasets/outdoor/train/haze/', help='!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
parser.add_argument('--category', type=str, default='outdoor',help='!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
####
parser.add_argument('--hazy_test_dir', type=str, default='./datasets/outdoor/test/haze/',help='!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
parser.add_argument('--clear_test_dir', type=str, default='./datasets/outdoor/test/clear/',help='!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
########
parser.add_argument('--save_model_dir', default='./checkpoints/outdoor/',help='Directory to save the model')
parser.add_argument('--save_good_model_dir', default='./checkpoints/GoodT/',help='Directory to save the model')
parser.add_argument('--save_bad_model_dir', default='./checkpoints/BadT/',help='Directory to save the model')
# parser.add_argument('--save_val_dir', default='./SRplus_val_result/',help='Directory to save the val results')
parser.add_argument('--save_val_dir', default='./results/outdoor_val/',help='Directory to save the val results')
# parser.add_argument('--log_dir', default='./logs',help='Directory to save the logs')
parser.add_argument('--log_dir', default='./logs/train_log.txt',help='Directory to save the logs')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--max_epoch', type=int, default=100)
parser.add_argument('--decay_epoch', type=int, default=50)
parser.add_argument('--start_epoch', type=float, default=1)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--gps', type=int, default=3)
parser.add_argument('--blocks', type=int, default=6)#设置去雾模块
# parser.add_argument('--blocks', type=int, default=3)
parser.add_argument('--l1_weight', type=float, default=1.0)
parser.add_argument('--perceptual_weight', type=float, default=0.1)
parser.add_argument('--n_threads', type=int, default=0)
args = parser.parse_args('')

if __name__ == "__main__":

    transforms_train = [
        transforms.ToTensor()  # range [0, 255] -> [0.0,1.0]
    ]

    train_dataset = TestDataloader(args.hazy_dir,args.clear_dir,transform=transforms_train)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.n_threads)

    #测试
    val_sets = TestDataloader(args.hazy_test_dir, args.clear_test_dir, transform=transforms_train)
    val_loader = DataLoader(dataset=val_sets, batch_size=args.batch_size // args.batch_size, shuffle=False)

    dataset_length = len(train_loader)

    logger_train=utils.Logger(args.max_epoch,dataset_length)
    #测试
    logger_val = utils.Logger(args.max_epoch, len(val_loader))

    T = SRNetPlus.SRNet(gps=args.gps,blocks=args.blocks).to(device)

    print('The models are initialized successfully!')

    T.train()

    total_params = sum(p.numel() for p in T.parameters() if p.requires_grad)
    print("Total_params: ==> {}".format(total_params))

    opt_T = optim.Adam(T.parameters(), lr=args.lr, betas=(0.5, 0.999))

    lr_scheduler_T = torch.optim.lr_scheduler.LambdaLR(opt_T, lr_lambda=utils.LambdaLR(args.max_epoch, args.start_epoch, args.decay_epoch).step)

    loss_l1 = nn.L1Loss().to(device)   # L1 loss
    loss_per = VGG19PerceptualLoss.PerceptualLoss().to(device)   # perceptual loss

    max_ssim = 0
    max_psnr = 0
    all_ssims = []
    all_psnrs = []

    for epoch in range(args.start_epoch, args.max_epoch + 1):

        ssims = []  # 每轮清空
        psnrs = []  # 每轮清空

        start_time = time.time()

        for i, batch in enumerate(train_loader):

            x = batch[0].to(device)   # clear images
            y = batch[1].to(device)

            output = T(x)

            loss_L1 =  loss_l1(output, y) * args.l1_weight
            loss_Per = loss_per(output, y) * args.perceptual_weight

            loss = loss_L1 + loss_Per

            opt_T.zero_grad()
            loss.backward()
            opt_T.step()

            psnr1 = psnr(output, y)
            ssim1 = ssim(output, y).item()

            psnrs.append(psnr1)
            ssims.append(ssim1)

            logger_train.log_train({},images={'input': x, 'Output': output})
            ###################
            sys.stdout.write(
                '\rEpoch %03d/%03d [%04d/%04d] -- Loss %.6f --Max_PSNR：%.6f --Max_SSIM：%.6f' % (
                epoch, args.max_epoch, i + 1, dataset_length, loss.item(), max_psnr, max_ssim))
        ##########

        # one_epoch_time = time.time() - start_time
        # psnr_eval = np.mean(psnrs)
        # ssim_eval = np.mean(ssims)
        #
        # if psnr_eval > max_psnr:
        #
        #     max_psnr = max(max_psnr, psnr_eval)
        #
        #     torch.save(T.state_dict(), args.save_bad_model_dir + args.category + "_T.pth")
        #
        # if ssim_eval > max_ssim:
        #
        #     max_ssim = max(max_ssim, ssim_eval)
        #############################
        #
        #     torch.save(T.state_dict(), args.save_model_dir + args.category + "T_Best_SSIM.pth")

        lr_scheduler_T.step()

        # utils.print_log(epoch,args.max_epoch,one_epoch_time=one_epoch_time,val_psnr=psnr_eval,val_ssim=ssim_eval)

        ######################################测试###################################

        with torch.no_grad():

            T.eval()

            torch.cuda.empty_cache()

            images_val = []
            images_name = []
            print("epoch:{}---> Metrics are being evaluated！".format(epoch))

            for a, batch_val in enumerate(val_loader):

                haze_val  = batch_val[0].to(device)
                clear_val = batch_val[1].to(device)

                image_name= batch_val[2][0]

                output_val = T(haze_val)

                images_val.append(output_val)
                images_name.append(image_name)

                psnr1 = psnr(output_val, clear_val)
                ssim1 = ssim(output_val, clear_val).item()

                psnrs.append(psnr1)
                ssims.append(ssim1)

                logger_val.log_val({'PSNR': psnr1,
                                    'SSIM': ssim1},
                                   images={'output_val': output_val, 'val': clear_val})


            psnr_eval = np.mean(psnrs)
            ssim_eval = np.mean(ssims)

            if psnr_eval > max_psnr:

                max_psnr = max(max_psnr, psnr_eval)

                torch.save(T.state_dict(), args.save_model_dir + args.category + "_Best_PSNR.pth")

                for i in range(len(images_name)):
                    torchvision.utils.save_image(images_val[i], args.save_val_dir + "{}".format(images_name[i]))

            if ssim_eval > max_ssim:

                max_ssim = max(max_ssim, ssim_eval)

                torch.save(T.state_dict(), args.save_model_dir + args.category + "_Best_SSIM.pth")









