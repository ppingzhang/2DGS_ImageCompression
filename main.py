
import torch

import os
import torch.nn as nn
from torch.optim import Adam
import argparse
import logging

from GS import generate_2D_gaussian_splatting
from utils import *
from dataset import *
from loss import *
import time
from tqdm import tqdm

from VQ import VectorQuantizerKMeans
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='2D Gaussian Splatting for Video')
    parser.add_argument('--image_height', default=512//2, type=int, help='image_height')#512
    parser.add_argument('--image_width', default=768//2, type=int, help='image_width')  # 768
    
    parser.add_argument('--patch_h_number', default=1, type=int, help='patch_h_number')#4
    parser.add_argument('--patch_w_number', default=1, type=int, help='patch_w_number')#6
    parser.add_argument('--num_embeddings', default=64, type=int, help='num_embeddings')
 

    parser.add_argument('--num_epochs', default=2000, type=int, help='num_epochs')
    parser.add_argument('--primary_samples', default=100, type=int, help='primary_samples')
    parser.add_argument('--backup_samples', default=100, type=int, help='backup_samples')
    parser.add_argument('--densification_interval', default=200, type=int, help='num_epochs')

    parser.add_argument('--gradient_threshold', default=0.0005, type=float, help='gradient_threshold')
    parser.add_argument('--gaussian_threshold', default=0.0005, type=float, help='gaussian_threshold')
    parser.add_argument('--gaussian_remove_threshold', default=0.001, type=float, help='gaussian_remove_threshold')
    parser.add_argument('--scale_reduction_factor', default=1.6, type=float, help='scale_reduction_factor')

    parser.add_argument('--display_interval', default=100, type=int, help='display_interval')
    parser.add_argument('--quantied_bits', default=8, type=int, help='quantied_bits')

    parser.add_argument('--device', default="cuda", type=str, help='device')
    parser.add_argument('--lr', default="0.01", type=float, help='learning rate')
    parser.add_argument('--display_loss', default=False, type=bool, help='display_loss')
    
    parser.add_argument('--clip_gradient', default="100", type=float, help='clip_gradient')

    parser.add_argument('--eval', default=False, type=bool, help='eval')

    parser.add_argument('--image_dir', default="./dta/kodim01.png", type=str, help='image_dir')
    parser.add_argument('--outf', default="output", type=str, help='output direction')

    
    parser.add_argument('--seed', default=1, type=int, help='seed')
    parser.add_argument('--up_ratio', default=2, type=int, help='up_ratio')
 
 
    args = parser.parse_args()

    args.num_samples = args.primary_samples + args.backup_samples
    
    coeff = 0.00001
    # set random seed
    set_random_seed(args.seed)
    img_name = args.image_dir.split('/')[-1].split('.png')[0]
    print(f'Current image is {img_name}')
    label = f"PS_{args.patch_h_number}_{args.patch_w_number}_SN_{args.primary_samples}_{args.backup_samples}_epoch_{args.num_epochs}/{img_name}/"
    directory = f"./output/{args.outf}/{label}"
    os.makedirs(directory, exist_ok=True)
    
    if not args.eval:
        logging.basicConfig(filename=f'{directory}/model_log.txt', filemode='w', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logger = logging.getLogger()
        logger.info("Command Line Arguments:")
        for arg, value in vars(args).items():
            logger.info("%s: %s", arg, value)
    
    # load image
    data_dict = load_img(args, args.up_ratio)
    args.image_height = data_dict['height']
    args.image_width = data_dict['width']
    target_tensor = data_dict['img']
    or_target_tensor = data_dict['or_img']

    # Initilize the parameters
    sigma_values    = torch.logit(torch.rand((args.num_samples, 2), device=args.device)* 0.1)
    rotation_angles = torch.logit(torch.rand((args.num_samples, 1), device=args.device))
    rgbas           = torch.rand((args.num_samples, 4), device=args.device)
    rgbas[:, 3]     = (rgbas[:, 3] + 0.1) / 1.1  # avoid fully transparent initial alphas
    rgbas           = torch.logit(rgbas)
    positions       = torch.logit(torch.rand((args.num_samples, 2), device=args.device))
   
    W = nn.Parameter(torch.cat([sigma_values, rotation_angles, rgbas, positions], -1))

    vqk = VectorQuantizerKMeans(torch.cat([sigma_values, rotation_angles], -1), k=args.num_embeddings)
    vqk_rgbas = VectorQuantizerKMeans(rgbas, k=args.num_embeddings)

    starting_size   = args.primary_samples
    left_over_size  = args.backup_samples
    persistent_mask = torch.cat([torch.ones(starting_size, dtype=bool), torch.zeros(left_over_size, dtype=bool)], dim=0)
    current_marker  = starting_size

    
    other_params = []
    other_params.append(W)
    optimizer = Adam(other_params, lr=args.lr)

    loss = torch.zeros([3, 2])
    loss_history = []
    psnr_history = []
    psnr_up_history = []
    ssim_up_history = []
    bpp_loss_history = []
    
    start_epoch = 0
    best_psnr = 0
    common_indices = []
    distinct_indices = []
    
    if os.path.exists(f'{directory}/best.pth'):
        
        checkpoint = torch.load(f'{directory}/best.pth')
        W.data = checkpoint["w_values"]
        persistent_mask = checkpoint["persistent_mask"]
        start_epoch = checkpoint["epoch"] + 1

        vqk.centroids = checkpoint['centroids']
        vqk.labels = checkpoint['label']

        vqk_rgbas.centroids = checkpoint['centroids_rgba']
        vqk_rgbas.labels = checkpoint['label_rgba']
        
        psnr_history = checkpoint["psnr_history"]
        best_psnr =  checkpoint["best_psnr"]
        print(f'----load---model {start_epoch}')
    
    training_start_time = time.time()
    if not args.eval: 
        with tqdm(initial=start_epoch, total=args.num_epochs) as pbar:
            
            for epoch in range(start_epoch, args.num_epochs):
                
                #check_for_nan({'loss':loss})
                epoch_start_time = time.time()
                # Gassians removed
                if epoch % (args.densification_interval + 1) == 0 and epoch > 0:
                    indices_to_remove = (torch.sigmoid(W[:, 3]) < args.gaussian_remove_threshold).nonzero(as_tuple=True)[0]
                    if len(indices_to_remove) > 0:
                        logger.info(f"number of pruned points: {len(indices_to_remove)}\n")

                    persistent_mask[indices_to_remove] = False
                    W.data[~persistent_mask] = 0.0

                bn_sigma_values    = torch.sigmoid(W[persistent_mask, :2])
                bn_rotation_angles = torch.sigmoid(W[persistent_mask, 2:3])
                bn_rgbas           = torch.sigmoid(W[persistent_mask, 3:7])
                bn_positions       = torch.sigmoid(W[persistent_mask, 7:9])
                
                if epoch % 200==0:
                    save_idx = True
                else:
                    save_idx = False
                sigma_rotation_quantized, sigma_rotation_indices, sigma_rotation_commit_loss = vqk(torch.cat([bn_sigma_values, bn_rotation_angles], -1), save=save_idx)
                rgba_quantized, rgba_indices, rgba_commit_loss = vqk_rgbas(bn_rgbas, save=save_idx)
                    

                zero_indices = sigma_rotation_quantized == 0
                sigma_rotation_quantized[zero_indices] += coeff

                rgba_zero_indices = rgba_quantized == 0
                rgba_quantized[rgba_zero_indices] += coeff
                #rgba_quantized = torch.sigmoid(color_model(base_rgba_quantized) + base_rgba_quantized)
                g_tensor_batch = generate_2D_gaussian_splatting(args.image_height, args.image_width, args.patch_h_number, args.patch_w_number, sigma_rotation_quantized[:,:2], sigma_rotation_quantized[:,2:3], rgba_quantized, bn_positions)
                
                up_output_tensor = F.interpolate(g_tensor_batch.unsqueeze(0).permute(0, 3, 1, 2), scale_factor=args.up_ratio, mode='bicubic', align_corners=False).permute(0, 2, 3, 1)[0]
                
                #['l1_ssim_1_2', 'l1_ssim_1', 'l1_ssim_2', 'l1_1', 'l1_2', 'l1_1_2']:
                loss = combined_loss(up_output_tensor, or_target_tensor) + combined_loss(g_tensor_batch, target_tensor) 
                
                psnr_v = psnr(g_tensor_batch, target_tensor)
                ssim_v = ssim(up_output_tensor, or_target_tensor)
                psnr_v_up = psnr(up_output_tensor, or_target_tensor)
                optimizer.zero_grad()
                loss.backward()
                

                torch.nn.utils.clip_grad_norm_(W, args.clip_gradient)

                # clean gradient
                if persistent_mask is not None:
                    W.grad.data[~persistent_mask] = 0.0

                if epoch % args.densification_interval == 0 and epoch > 0:

                    # Calculate the norm of gradients
                    common_indices, distinct_indices = clone_split_Gaussian(args, W, persistent_mask)
                
                    # Split points with large coordinate gradient and large gaussian values and descale their gaussian
                    if len(common_indices) > 0:
                        if current_marker + len(common_indices) < args.num_samples:
                            start_index = current_marker + 1
                            end_index = current_marker + 1 + len(common_indices)
                            persistent_mask[start_index: end_index] = True
                            W.data[start_index:end_index, :] = W.data[common_indices, :]
                            W.data[start_index:end_index, 0:2] /= args.scale_reduction_factor
                            W.data[common_indices, 0:2] /= args.scale_reduction_factor
                            current_marker = current_marker + len(common_indices)

                    # Clone it points with large coordinate gradient and small gaussian values
                    if len(distinct_indices) > 0:
                        if current_marker + len(distinct_indices) < args.num_samples:
                            start_index = current_marker + 1
                            end_index = current_marker + 1 + len(distinct_indices)
                            persistent_mask[start_index: end_index] = True
                            W.data[start_index:end_index, :] = W.data[distinct_indices, :]
                            current_marker = current_marker + len(distinct_indices)

                optimizer.step()
                loss_history.append(loss.item())
                
                #bpp_loss_history.append(bpp_loss.item())
                #vq.embeddings.weight.data = torch.clamp(vq.embeddings.weight.data, 0+coeff, 1-coeff)
                    
                if best_psnr < psnr_v_up.item():
                    best_psnr = psnr_v_up.item()
                    torch.save({
                        'w_values': W,
                        'epoch': epoch,
                        'centroids':vqk.centroids,
                        'label':vqk.labels,
                        'centroids_rgba':vqk_rgbas.centroids,
                        'label_rgba':vqk_rgbas.labels,
                        "persistent_mask": persistent_mask,
                        "psnr_history":psnr_history,
                        "best_psnr": best_psnr
                    }, f'{directory}/best.pth')
                    
                if epoch % args.display_interval == 0 or epoch == args.num_epochs-1:
                    
                    psnr_history.append(psnr_v.item())
                    psnr_up_history.append(psnr_v_up.item())
                    #ssim_up_history.append(ssim_v.item())
                    
                    if epoch > 4000:
                        polt_fig(args, g_tensor_batch, target_tensor, psnr_history, directory, epoch)
                    
                    current_lr = optimizer.param_groups[0]['lr']

                    torch.save({
                        'w_values': W,
                        'epoch': epoch,
                        'centroids': vqk.centroids,
                        'label':vqk.labels,
                        'centroids_rgba':vqk_rgbas.centroids,
                        'label_rgba':vqk_rgbas.labels,
                        "persistent_mask": persistent_mask,
                        "psnr_history": psnr_history,
                        "best_psnr": best_psnr
                    }, f'{directory}/lastest.pth')

                pbar.update(1)
                epoch_ending_time = time.time()
                pbar.set_postfix({"time": epoch_ending_time-epoch_start_time, "epoch":epoch, "Loss": loss.item(), "sigma_commit_loss": sigma_rotation_commit_loss.item(), "PSNR": psnr_v.item(),"PSNR_U": psnr_v_up.item(),"split_point":len(common_indices), "cloned points": len(distinct_indices), "points": W[persistent_mask].shape[0]})
                logger.info(f"time:{epoch_ending_time-epoch_start_time} epoch:{epoch} Loss:{loss.item()} sigma_commit_loss:{sigma_rotation_commit_loss.item()} PSNR:{psnr_v.item()} PSNR_U:{psnr_v_up.item()} split_point:{len(common_indices)} cloned points:{len(distinct_indices)} points: {W[persistent_mask].shape[0]}")
                logger.handlers[0].flush()

    with torch.no_grad():
        
        print(f'point number:{persistent_mask.sum()}, shape:{persistent_mask.shape}')
        
        num_point = persistent_mask.sum()
        bit_n = int(np.log2(num_point))
        
        best_pnsr = 0.0
        best_ssim = 0.0
        best_bpp = 0.0
        best_ln = 0.0
        de_ = 0.0
        vqk_c = vqk.centroids.data.clone()
        vqk_rgbas_c = vqk_rgbas.centroids.data.clone()
        
        for bit_length in [bit_n, bit_n+1, bit_n+2, bit_n+3, bit_n+4, bit_n+6]:
            args.quantied_bits = bit_length
            
            vqk.centroids.data = vqk_c
            vqk_rgbas.centroids.data = vqk_rgbas_c
        
            psnr_v, ssim_v, bpp_v = cal_bpp_color_kmeans_up(args, W[persistent_mask, :], or_target_tensor, vq=vqk, vq_color=vqk_rgbas, coding_dir=directory, image_name=img_name, save_dir=directory)
    
            de_cur = psnr_v/(bpp_v)
            if de_cur > de_:
                best_pnsr = psnr_v
                best_ssim = ssim_v
                best_bpp = bpp_v
                best_ln = bit_length
                de_ = de_cur
            print(f"{de_cur} best_bit length:{best_ln}, best_pnsr:{psnr_v}, best_ssim:{best_ssim}, best_bpp:{bpp_v}" )
        print(f"best_bit length:{best_ln}, best_pnsr:{best_pnsr}, best_ssim:{best_ssim}, best_bpp:{best_bpp}" )