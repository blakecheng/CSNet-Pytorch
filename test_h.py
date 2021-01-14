import argparse
import torch
from torch.autograd import Variable
import numpy as np
import time, math, glob
import scipy.io as sio
from lib.network import CSNet,CSNet_Enhanced
import os
from lib.Hierarchical_network import HierarchicalCSNet
from lib.variation_hierachical_network import VariationHierarchicalCSNet

parser = argparse.ArgumentParser(description="PyTorch LapSRN Eval")
# parser.add_argument("--cuda", action="store_true", help="use cuda?")
parser.add_argument("--model", default="experiment/hirerachical/epochs_subrate_0.1_blocksize_32/net_epoch_0_0.014720.pth", type=str, help="model path")
parser.add_argument("--dataset", default="Test/Set5_mat", type=str, help="dataset name, Default: Set5")
# parser.add_argument('--block_size', default=32, type=int, help='CS block size')
# parser.add_argument('--sub_rate', default=0.1, type=float, help='sampling sub rate')
parser.add_argument('--mt',default='HierarchicalCSNet')


def PSNR(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)

opt = parser.parse_args()
cuda = torch.cuda.is_available()



# if cuda and not torch.cuda.is_available():
#     raise Exception("No GPU found, please run without --cuda")


if opt.mt == "HierarchicalCSNet":
    args = torch.load(os.path.split(opt.model)[0]+"/opt.pt")
    use_variance_estimation = True
    model = HierarchicalCSNet(args.block_size, args.sub_rate,group_num=args.group_num,mode=args.fusion_mode,variance_estimation=use_variance_estimation,z_channel=args.zc)
elif opt.mt == "VariationHierachicalCSNet":
    args = torch.load(os.path.split(opt.model)[0]+"/opt.pt")
    use_variance_estimation=True
    model = VariationHierarchicalCSNet(args.block_size, args.sub_rate,group_num=args.group_num,mode=args.fusion_mode)




model.load_state_dict(torch.load(opt.model))


image_list = glob.glob(opt.dataset+"/*.*") 

avg_psnr_predicted_list = [0.0 for _ in range(args.group_num)]
avg_elapsed_time = 0.0

with torch.no_grad():
    for image_name in image_list:
        print("Processing ", image_name)
        im_gt_y = sio.loadmat(image_name)['im_gt_y']

        im_gt_y = im_gt_y.astype(float)

        im_input = im_gt_y/255.

        im_input = Variable(torch.from_numpy(im_input).float()).view(1, -1, im_input.shape[0], im_input.shape[1])

        if cuda:
            model = model.cuda()
            im_input = im_input.cuda()
        else:
            model = model.cpu()

        start_time = time.time()
        if use_variance_estimation:
            res_list,var_list,latent_list = model(im_input)
        else:
            res_list = model(im_input)
        elapsed_time = time.time() - start_time
        avg_elapsed_time += elapsed_time

        for i in range(args.group_num):
            res = res_list[i].cpu()
            im_res_y = res.data[0].numpy().astype(np.float32)

            im_res_y = im_res_y*255.
            im_res_y[im_res_y<0] = 0
            im_res_y[im_res_y>255.] = 255.
            im_res_y = im_res_y[0,:,:]

            psnr_predicted = PSNR(im_gt_y, im_res_y,shave_border=0)
            print(psnr_predicted)
            avg_psnr_predicted_list[i] += psnr_predicted

    print("Dataset=", opt.dataset)
    PSNR_RESULT = [avg_psnr_predicted/len(image_list) for avg_psnr_predicted in avg_psnr_predicted_list]
    save_path = opt.model[:-4]+"_result.txt"
    fileObject = open(save_path, 'w')
    for ip in PSNR_RESULT:
        fileObject.write(str(ip))
        fileObject.write('\n')
    fileObject.close()
    print("PSNR_predicted=",PSNR_RESULT)
    print("It takes average {}s for processing".format(avg_elapsed_time/len(image_list)))
