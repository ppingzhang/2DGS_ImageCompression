from PIL import Image 
import numpy as np
import torch

import os
import torch.nn as nn
import imageio

import torch
import shutil
from torch.optim import Adam
import torch.nn.functional as F
import matplotlib.pyplot as plt

from datetime import datetime
import torchac
import zlib
import torchvision.transforms as transforms
import imageio
from vector_quantize_pytorch import VectorQuantize

import bz2
import gzip
import zipfile
import lzma
import math
import GS
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from loss import *

from pytorch_msssim import MS_SSIM



def select_grid_point(sub_coords, all_coords, selected_tensor):
    comparison = torch.eq(sub_coords[:, None], all_coords.unsqueeze(0))
    matching_indices = torch.all(comparison, dim=-1)
    indices = torch.nonzero(matching_indices, as_tuple=False)
    #assert indices[:,0] == indices[:, 1]
    result = selected_tensor[indices[:, 1]]
    return result
    
    
## saving process
def save_args_to_file(args, filename):
    with open(filename, 'w') as file:
        for arg in vars(args):
            file.write(f"{arg}: {getattr(args, arg)}\n")

## set random seed:
def set_random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

# clone_split_Gaussian:
def clone_split_Gaussian(args, W, persistent_mask):
    # Calculate the norm of gradients
    gradient_norms = torch.norm(W.grad[persistent_mask][:, 7:9], dim=1, p=2)
    gaussian_norms = torch.norm(torch.sigmoid(W.data[persistent_mask][:, 0:2]), dim=1, p=2)

    sorted_grads, sorted_grads_indices = torch.sort(gradient_norms, descending=True)
    sorted_gauss, sorted_gauss_indices = torch.sort(gaussian_norms, descending=True)
    
    large_gradient_mask = (sorted_grads > args.gradient_threshold)
    large_gradient_indices = sorted_grads_indices[large_gradient_mask]

    large_gauss_mask = (sorted_gauss > args.gaussian_threshold)
    large_gauss_indices = sorted_gauss_indices[large_gauss_mask]

    common_indices_mask = torch.isin(large_gradient_indices, large_gauss_indices)
    common_indices = large_gradient_indices[common_indices_mask]
    distinct_indices = large_gradient_indices[~common_indices_mask]
    return common_indices, distinct_indices

def perfome_clone_split(args, W, persistent_mask, common_indices, distinct_indices, current_marker):
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

    return W, current_marker, persistent_mask

## checkinf
def replace_inf_with_max(tensor, **kargs):
    # Check for infinity values
    has_inf = torch.isinf(tensor)

    # If there are infinity values, replace them with the maximum finite value
    if has_inf.any():
        max_finite_value = torch.max(tensor[~has_inf])
        #tensor[has_inf] = max_finite_value
        tensor = torch.where(has_inf, max_finite_value, tensor)
        #for kk in kargs:
        #    print(f"{kk}:{kargs[kk].max()}, {kargs[kk].min()}" )
        #print(tensor.max(), tensor.min())
        #tensor = torch.clamp(tensor, max=1000)
        #print(tensor.max())
    return tensor

def replace_nan_with_zero(W):
    if W.grad is not None:
        W.grad[W.grad != W.grad] = 0
    return W

# entropy coding
def arithmetic_encoding(x, cdf=None, sym=None):
    """ Arithmetic encoding for a tensor with torchac. """    
    with torch.no_grad():
        x = x.detach().contiguous().view(-1).cpu()

        sym_, inverse_, counts_ = x.unique(return_inverse=True, return_counts=True)
        inverse_ = inverse_.to(torch.int16)
        counts_ = torch.cat([torch.zeros([1], dtype=torch.int64, device=counts_.device), counts_])
        cdf_ = torch.cumsum(counts_, dim=0).float() / counts_.sum().float()
        
        #print(inverse_, '---')
        if sym is None and cdf is None:
            sym, inverse, counts = x.unique(return_inverse=True, return_counts=True)
            inverse = inverse.to(torch.int16)
            counts = torch.cat([torch.zeros([1], dtype=torch.int64, device=counts.device), counts])
            cdf = torch.cumsum(counts, dim=0).float() / counts.sum().float()
        else:
            inverse = x.to(torch.int16)
        #print(sym[:10], sym_[:10])
        #print(sym.max(), sym_.max())
        
        
        byte_stream = torchac.encode_float_cdf(cdf[None].repeat(np.prod(x.shape), 1), inverse, check_input_bounds=True, needs_normalization=True)        
        inverse_out = torchac.decode_float_cdf(cdf[None].repeat(np.prod(x.shape), 1), byte_stream).int()
        assert inverse_out.int().equal(inverse.int())
        x_out = sym[inverse_out.long()]
        
        assert x_out.long().equal(x.long())
        return x_out, sym, cdf, byte_stream

ms_ssim_module = MS_SSIM(data_range=1.0, size_average=True, channel=3)
def cal_bpp_color_kmeans_up(args, W, target_tensor, vq=None, vq_color=None, coding_dir='./', compress_type='lzma', up_ratio=2, image_name="", save_dir=""): # # lzma bz2 gzip
    
    os.makedirs(save_dir, exist_ok=True)
    quant_result = {}
    bit_dict = {}
    #print(vq.centroids.shape, vq.centroids.shape, '+dd+++')
    coding_bits = 0.0
    sigma_values    = torch.sigmoid(W[:,  :2])
    rotation_angles = torch.sigmoid(W[:, 2:3])
    rgbas           = torch.sigmoid(W[:, 3:7])
    positions       = torch.sigmoid(W[:, 7:9])
    
    if not vq is None:
        sigma_values_, sigma_indices_, _ = vq(torch.cat([sigma_values, rotation_angles], -1), assign=True)
    if not vq_color is None:
        rgbas_, rgbas_indices_, _ = vq_color(rgbas)
    #print(sigma_values_.shape, '---')
    tensor_batch = GS.generate_2D_gaussian_splatting(args.image_height, args.image_width, args.patch_h_number, args.patch_w_number, sigma_values_[:, :2], sigma_values_[:,2:3], rgbas_, positions)
    tensor_batch = F.interpolate(tensor_batch.unsqueeze(0).permute(0, 3, 1, 2), scale_factor=up_ratio, mode='bicubic', align_corners=False).permute(0, 2, 3, 1)[0]
    
    or_psnr_v = psnr(tensor_batch, target_tensor)
    ssim_v = 1 - ms_ssim_module(tensor_batch.unsqueeze(0).permute(0, 3, 1, 2), target_tensor.unsqueeze(0).permute(0, 3, 1, 2))
    
    # here we fix the symbol book or you can use the adaptive symbol book.
    # we set a uniform distribution for symbols
    sym_range = 2**(args.quantied_bits) +1# #256+1
    symbol_tensor = torch.arange(sym_range)
    prob_per_symbol = 1 / sym_range
    prob_tensor = torch.full((sym_range,), prob_per_symbol)
    cdf_tensor = torch.cumsum(prob_tensor, dim=0)
    
    
    ps_sym_range = 2**10 +1# #256+1
    ps_symbol_tensor = torch.arange(ps_sym_range)
    ps_prob_per_symbol = 1 / ps_sym_range
    ps_prob_tensor = torch.full((ps_sym_range,), ps_prob_per_symbol)
    ps_cdf_tensor = torch.cumsum(ps_prob_tensor, dim=0)
    
    
    #print(f"orgain PSNR:{or_psnr_v}, ssim: {ssim_v}")
    if not vq_color is None:
        quant_kmeans_v_color, new_kmeans_color = quant_tensor(vq_color.centroids.data, args.quantied_bits)
        
        _, sym_color, cdf_color, bits_length_color = arithmetic_encoding(new_kmeans_color['quant'], cdf_tensor, symbol_tensor)
        value_meta_rgba = {#'sym': sym,
            #'cdf': cdf_color,
            'bits_length': bits_length_color,
            'offset': [new_kmeans_color['min'], new_kmeans_color['scale']]
        }
        
        quant_result["q_rgba"] = value_meta_rgba
        coding_bits += len(bits_length_color)
        bit_dict["q_rgba"] = len(bits_length_color)
        vq_color.centroids.data = quant_kmeans_v_color

        q_alpha_colour, q_rgbas_indices, _ = vq_color(rgbas, assign=True)

        _, sym_rgbas, cdf_rgbas, bits_length_rgbas = arithmetic_encoding(q_rgbas_indices, cdf_tensor, symbol_tensor)
        value_meta_rgba = {#'sym': sym,
            #'cdf': cdf_rgbas,
            'bits_length': bits_length_rgbas
            #'offset': [q_sigma_indice_dict['min'], q_sigma_indice_dict['scale']]
        }
        coding_bits += len(bits_length_rgbas)
        quant_result["rgb_indices"] = value_meta_rgba
        bit_dict["rgb_indices"] = len(bits_length_rgbas)
    

    q_pixel, q_pixel_dict  = quant_tensor(positions, bits=10)
    #q_rotation, q_rotation_dict  = quant_tensor(rotation_angles, bits=args.quantied_bits)
    
    if not vq is None:
        quant_kmeans_v, new_kmeans = quant_tensor(vq.centroids.data, args.quantied_bits)
        
        _, sym, cdf, bits_length = arithmetic_encoding(new_kmeans['quant'], cdf_tensor, symbol_tensor)
        value_meta = {#'sym': sym,
            #'cdf': cdf,
            'bits_length': bits_length,
            'offset': [new_kmeans['min'], new_kmeans['scale']]
        }
        quant_result["sigma_k"] = value_meta
        coding_bits += len(bits_length)
        bit_dict["sigma_k"] = len(bits_length)
        vq.centroids.data = quant_kmeans_v

        q_sigma, q_sigma_indices, _ = vq(torch.cat([sigma_values, rotation_angles], -1), assign=True)

        #q_sigma_indice, q_sigma_indice_dict  = quant_tensor(q_sigma_indices, bits=7)
        #print(q_sigma_indices.min(), q_sigma_indices.max(),  '---')
        _, sym, cdf, bits_length = arithmetic_encoding(q_sigma_indices)
        value_meta = {#'sym': sym,
            #'cdf': cdf,
            'bits_length': bits_length
            #'offset': [q_sigma_indice_dict['min'], q_sigma_indice_dict['scale']]
        }
        coding_bits += len(bits_length)
        quant_result["sigma_indices"] = value_meta
        bit_dict["sigma_indices"] = len(bits_length)
       
    #check_for_nan({"q_sigma":q_sigma, "q_rotation":q_rotation, "q_alpha_colour":q_alpha_colour, "q_pixel":q_pixel, "q_rotation":q_rotation, "q_alpha_colour":q_alpha_colour, "q_pixel":q_pixel})
    q_tensor_batch = GS.generate_2D_gaussian_splatting(args.image_height, args.image_width, args.patch_h_number, args.patch_w_number, q_sigma[:, :2], q_sigma[:, 2:3], q_alpha_colour, q_pixel)
    q_tensor_batch = F.interpolate(q_tensor_batch.unsqueeze(0).permute(0, 3, 1, 2), scale_factor=up_ratio, mode='bicubic', align_corners=False).permute(0, 2, 3, 1)[0]
    #print(image_name, '=====')
    # save image 
    Image.fromarray(np.uint8(q_tensor_batch.cpu().detach().numpy()*255.0)).save(f"{save_dir}/{image_name}.png")
    #print(f"The decoded image has been save to {save_dir}/{image_name}")
    #print(q_tensor_batch.shape, sr_img.shape)


    psnr_v = psnr(q_tensor_batch, target_tensor)
    ssim_v = 1 - ms_ssim_module(q_tensor_batch.unsqueeze(0).permute(0, 3, 1, 2), target_tensor.unsqueeze(0).permute(0, 3, 1, 2))
    #print(f"quantized PSNR:{psnr_v} SSIM:{ssim_v }")
    
    #temp_dir = './tmp_result/'
    
    #print('--arthimatic--')
    
    quant_ckt = {
                 'q_pixel': q_pixel_dict,
                 }
    
    for k, layer_wt in quant_ckt.items():
        _, sym, cdf, bits_length = arithmetic_encoding(layer_wt['quant'], ps_cdf_tensor, ps_symbol_tensor)
        value_meta = {#'sym': sym,
            #'cdf': cdf,
            'bits_length': bits_length,
            'offset': [layer_wt['min'], layer_wt['scale']]
        }
        quant_result[k] = value_meta
        coding_bits += len(bits_length)

        bit_dict[k] = len(bits_length)
    
    # get the element name and its frequency
    ckpt_path = f'./{coding_dir}/model.pth.tar'
    compress_ckpt_path = f'{coding_dir}/model_compressed_{args.quantied_bits}.pth.tar'
    torch.save(quant_result, ckpt_path)

    # Compress the state_dict
    with open(ckpt_path, mode="rb") as f_in:
        with open(compress_ckpt_path, mode="wb") as f_out:
            f_out.write(zlib.compress(f_in.read(), zlib.Z_NO_COMPRESSION))

    #os.remove(ckpt_path)
    os.makedirs(f"{coding_dir}/zip/", exist_ok=True)
    #shutil.make_archive(f"{coding_dir}/zip/model_compressed_{args.quantied_bits}", 'zip', coding_dir)
    #z = zipfile.ZipFile('xin.tar.gz','w')

    compressed_filename = f"{coding_dir}/zip/model_compressed_{args.quantied_bits}.pth.tar.gz"
        
    if compress_type == 'lzma':
        #lzma_filename = f"{coding_dir}/zip/model_compressed_{args.quantied_bits}.pth.tar.gz"
        with open(compress_ckpt_path, 'rb') as tar_file:
            with lzma.open(compressed_filename, 'wb') as lzma_file:
                lzma_file.writelines(tar_file)
    
    
    total_bits = os.path.getsize(compressed_filename)*8
    ckpt_total_bits = os.path.getsize(f"{compress_ckpt_path}")*8
    # bits per pixel

    total_bpp = total_bits /(args.image_height * args.image_width)/up_ratio/up_ratio #
    #for k in bit_dict:
    #    print(f"{k}:{bit_dict[k]}")
    
    print(f'all coding bit:{coding_bits}, {coding_bits*8/(args.image_height * args.image_width*4)}')
    print(f"orgain PSNR:{or_psnr_v}, quantized PSNR:{psnr_v}, quantized SSIM:{ssim_v}")
    print(f'{args.image_height}x{args.image_width} quantied_bits:{args.quantied_bits} bits per pixel: {round(total_bpp, 4)}')
    print(f'{image_name} {args.quantied_bits} orgain PSNR:{or_psnr_v:.5f}, quantized PSNR:{psnr_v:.5f}, quantized SSIM:{ssim_v:.5f} bits per pixel: {total_bpp:.5f}')
    return psnr_v, ssim_v, round(total_bpp, 4)


def quant_model(model, quant_model_bit):
    model_list = [deepcopy(model)]
    #model_list = [model]
    if quant_model_bit == -1:
        return model_list, None
    else:
        cur_model = deepcopy(model)
        
        quant_ckt, cur_ckt = [cur_model.state_dict() for _ in range(2)]
        encoder_k_list = []
        for k,v in cur_ckt.items():
            new_v, quant_v = quant_tensor(v, quant_model_bit)
            quant_ckt[k] = quant_v
            cur_ckt[k] = new_v
            #print(k, '---',new_v, '---')
        
        cur_model.load_state_dict(cur_ckt)
        model_list.append(cur_model)
        
        return model_list, quant_ckt 

def quant_codebook(model, bits):
    
    cur_model = model
    
    quant_ckt, cur_ckt = [cur_model.state_dict() for _ in range(2)]
    encoder_k_list = []
    for k,v in cur_ckt.items():
        new_v,quant_v = quant_tensor(v, bits)
        quant_ckt[k] = quant_v
        cur_ckt[k] = new_v
    
    cur_model.load_state_dict(cur_ckt)
    
    return  cur_model, quant_ckt

################# Tensor quantization and dequantization #################
def quant_tensor(t, bits=8, eps=0.00001):
    tmin_scale_list = []
    # quantize over the whole tensor, or along each dimenstion
    t_min, t_max = t.min(), t.max()
    scale = (t_max - t_min) / (2**bits-1)
    tmin_scale_list.append([t_min, scale])
    for axis in range(t.dim()):
        t_min, t_max = t.min(axis, keepdim=True)[0], t.max(axis, keepdim=True)[0]
        if t_min.nelement() / t.nelement() < 0.02:
            scale = (t_max - t_min) / (2**bits-1)
            # tmin_scale_list.append([t_min, scale]) 
            tmin_scale_list.append([t_min.to(torch.float16), scale.to(torch.float16)]) 
    # import pdb; pdb.set_trace; from IPython import embed; embed() 
     
    quant_t_list, new_t_list, err_t_list = [], [], []
    for t_min, scale in tmin_scale_list:
        t_min, scale = t_min.expand_as(t), scale.expand_as(t)
        quant_t = ((t - t_min) / (scale+eps)).round().clamp(0, 2**bits-1)
        new_t = t_min + (scale+eps) * quant_t
        err_t = (t - new_t).abs().mean()
        quant_t_list.append(quant_t)
        new_t_list.append(new_t)
        err_t_list.append(err_t)   

    # choose the best quantization 
    best_err_t = min(err_t_list)
    best_quant_idx = err_t_list.index(best_err_t)
    best_new_t = new_t_list[best_quant_idx]
    if bits == 8:
        best_quant_t = quant_t_list[best_quant_idx].to(torch.uint8)
    else:
        best_quant_t = quant_t_list[best_quant_idx].to(torch.int16)
    best_tmin = tmin_scale_list[best_quant_idx][0]
    best_scale = tmin_scale_list[best_quant_idx][1]
    quant_t = {'quant': best_quant_t, 'min': best_tmin, 'scale': best_scale}
    
    return best_new_t, quant_t             

def save_tensor_image(tensor, file_path):
    tensor = tensor.clamp(0, 1)
    tensor = transforms.ToPILImage()(tensor)

    plt.imshow(tensor)
    plt.axis('off')
    plt.savefig(file_path, bbox_inches='tight', pad_inches=0) 
    plt.close()

def get_points_in_patches_(coords=None, h=1, w=1, hn=2, wn=2):
    
    norm_coords = (coords + 1.0) / 2.0
    b, c = norm_coords.shape
    sh = h / float(hn)
    sw = w / float(wn)
    
    idx_h = (norm_coords[:, 0] / sh).long().clamp(0, max=hn - 1)
    idx_w = (norm_coords[:, 1] / sw).long().clamp(0, max=wn - 1)
    dict_result = {f"{ii}_{jj}": [] for ii in range(hn) for jj in range(wn)}
    
    for kk in range(b):
        dict_result[f"{idx_h[kk]}_{idx_w[kk]}"].append(kk)
    
    return dict_result

def get_points_in_patches(coords=None, h=1, w=1, hn=2, wn=2):
    
    norm_coords = coords
    b, c = norm_coords.shape
    sh = h / float(hn)
    sw = w / float(wn)

    idx_h = (norm_coords[:, 0] / sh).long().clamp(0, max=hn - 1)
    idx_w = (norm_coords[:, 1] / sw).long().clamp(0, max=wn - 1)
    dict_result = {f"{ii}_{jj}": [] for ii in range(hn) for jj in range(wn)}
    #print(dict_result)
    for kk in range(b):
        dict_result[f"{idx_h[kk]}_{idx_w[kk]}"].append(kk)
    #print(dict_result)
    return dict_result

def create_gif(image_folder, gif_path):
    images = []
    file_names = sorted(os.listdir(image_folder)) 
    for filename in file_names:
        if filename.endswith('.png'):  
            file_path = os.path.join(image_folder, filename)
            images.append(imageio.imread(file_path))

    imageio.mimsave(gif_path, images, duration=0.5)  

def plot_points_with_radius(coordinates, save_path):
    
    fig, ax = plt.subplots()

    ax.scatter(coordinates[:, 0], coordinates[:, 1], color='blue', marker='o')

    ax.set_xlim(coordinates[:, 0].min() - 1, coordinates[:, 0].max() + 1)
    ax.set_ylim(coordinates[:, 1].min() - 1, coordinates[:, 1].max() + 1)

    ax.axis('off')

    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def check_for_nan(tensor_dict):
    for k in tensor_dict:
        if torch.isnan(tensor_dict[k]).any():
            str_ = ""
            for kk in tensor_dict:
                str_ += f"{kk}:{tensor_dict[kk].max()}, {tensor_dict[kk].min()}, "
                print(str_ ,"====")
            nan_positions = torch.nonzero(torch.isnan(tensor_dict[k]), as_tuple=False)
            print(nan_positions ,"====")
            
            raise ValueError(f"The {k} contains NaN values.")
    
def polt_fig( args, g_tensor_batch, target_tensor, loss_history, directory, epoch):
    num_subplots = 3 if args.display_loss else 2
    fig_size_width = 18 if args.display_loss else 12

    fig, ax = plt.subplots(1, num_subplots, figsize=(fig_size_width, 6))  

    ax[0].imshow(g_tensor_batch.cpu().detach().numpy())
    ax[0].set_title('2D Gaussian Splatting')
    ax[0].axis('off')

    ax[1].imshow(target_tensor.cpu().detach().numpy())
    ax[1].set_title('Ground Truth')
    ax[1].axis('off')

    if args.display_loss:
        ax[2].plot(range(0, epoch + 1, args.display_interval), loss_history[:epoch + 1])
        ax[2].set_title('PSNR vs. Epochs')
        ax[2].set_xlabel('Epoch')
        ax[2].set_ylabel('PSNR')
        ax[2].set_xlim(0, args.num_epochs)  # Set x-axis limits
    
    # Display the image
    #plt.show(block=False)
    plt.subplots_adjust(wspace=0.2)  # Adjust this value to your preference
    plt.pause(0.1)  # Brief pause
    generated_array = g_tensor_batch.cpu().detach().numpy()
    img = Image.fromarray((generated_array * 255).astype(np.uint8))
    # save generated image
    img.save(f"{directory}/{epoch}.jpg")
    # save figure 
    fig.savefig(f"{directory}/{epoch}.jpg")

    plt.clf()  # Clear the current figure
    plt.close()  # Close the current figure
  
def give_required_data(image_array, input_coords, image_size, device):
    # normalising pixel coordinates [-1,1]
    coords = torch.tensor(input_coords / [image_size[0], image_size[1]], device=device).float()
    center_coords_normalized = torch.tensor([0.5, 0.5], device=device).float()
    coords = (center_coords_normalized - coords) * 2.0
    #coords = coords * 2.0 - 1.0

    # Fetching the colour of the pixels in each coordinates
    #print(image_array.shape, input_coords)
    colour_values = [np.array(image_array[coord[0], coord[1],:]) for coord in input_coords]
    colour_values_np = np.array(colour_values)
    colour_values_tensor =  torch.tensor(colour_values_np, device=device).float()

    return colour_values_tensor, coords

def get_patch_centers(kh, kw):
    
    patch_h = 1.0 / kh
    patch_w = 1.0 / kw

    patch_centers = {}
    for i in range(kh):
        for j in range(kw):
            center_h = (i + 0.5) * patch_h
            center_w = (j + 0.5) * patch_w
            patch_centers[f"{i}_{j}"] =[center_h, center_w]
    return patch_centers

def visualize_points(points, heights, widths, save_path, color, alpha):
    
    #points =(points+1.0)/2.0
    fig, ax = plt.subplots()
    for point, w, h, c, a in zip(points, widths, heights, color, alpha):

        dc = np.append(c, a)
        #print(color)
        ellipse = Ellipse([point[1], point[0]], width=w, height=h, facecolor=dc)
        ax.add_patch(ellipse)

    ax.set_xlabel('Left-Right')
    ax.set_ylabel('Up-Down')

    ax.set_xlim([-1, 1])
    ax.set_ylim([1, -1])

    plt.gca().set_aspect('equal', adjustable='box')  
    #plt.grid(True)
    plt.title('Visualization of Points with Ellipses')
    plt.show()
    plt.savefig(save_path)

def find_circles_in_patch(patch, circle_centers, circle_radii):
    # Create tensor for patch corners
    patch_corners = torch.tensor([[patch[0], patch[1]], # (up, l)
                                  [patch[0] + patch[2], patch[1]], #[b, l]
                                  [patch[0], patch[1] + patch[3]], # [up, r]
                                  [patch[0] + patch[2], patch[1] + patch[3]]], #[b, r]
                                 device=circle_centers.device) 

    # Use torch.cdist for pairwise distance calculation
    distances = torch.cdist(patch_corners.view(1, -1, 2), circle_centers.view(-1, 1, 2)).squeeze()

    # Check if any distance is less than or equal to the radius for each circle
    is_inside = distances <= circle_radii.view(-1, 1)

    circle_center_inside_rect = (circle_centers[:, 0] >= patch[0]) & (circle_centers[:, 0] <= (patch[0] + patch[2])) & \
                                (circle_centers[:, 1] >= patch[1]) & (circle_centers[:, 1] <= (patch[1] + patch[3]))

    # Check if any corner is inside any circle
    included_circles = torch.any(torch.cat([is_inside, circle_center_inside_rect.unsqueeze(-1)], -1) , dim=1).nonzero().squeeze().tolist()
    return included_circles

def get_points_inlcude_overlap_in_patches(coords=None, signmax=None, h=1, w=1, hn=2, wn=2):
    # Assuming coords and signmax are torch tensors

    circle_centers = coords
    signmax[:, 0] /= hn
    signmax[:, 1] /= wn
    circle_radii = torch.min(signmax, dim=1).values

    results = {}
    for i in range(hn):
        for j in range(wn):
            patch = [i * (h / hn), j * (w / wn), (h / hn), (w / wn)]
            included_circles = find_circles_in_patch(patch, circle_centers, circle_radii)
            results[f"{i}_{j}"] = included_circles
            
    return results

def visualize_batch(images, save_path=None):
    # Convert the tensor to a NumPy array and extract the data
    images_np = images.detach().cpu().numpy()

    # Get the batch size
    batch_size, _, height, width = images_np.shape

    # Calculate the required number of rows and columns
    rows = int(np.sqrt(batch_size))
    cols = (batch_size + rows - 1) // rows

    # Create an image, initialized to zeros
    result_image = np.ones((3, rows * (height+10), cols * (width+10)))  # 3 represents RGB channels

    # Iterate over each sample, overlay it onto the result image
    for i in range(batch_size):
        row = i // cols
        col = i % cols
        try:
            result_image[:, row * height+(i*5):(row + 1) * height+(i*5), col * width+(i*5):(col + 1) * width+(i*5)] = images_np[i]
        except:
            pass
    # Clip the values of the result image to be between 0 and 1
    result_image = np.clip(result_image, 0, 1)

    # Display the result image
    plt.imshow(result_image.transpose((1, 2, 0)))  # Transpose channel order
    plt.axis('off')

    # Save the image to a file
    if save_path is not None:
        plt.savefig(save_path)

def remove_points(args, W, persistent_mask, logger):
    indices_to_remove = (torch.sigmoid(W[:, 3]) < args.gaussian_remove_threshold).nonzero(as_tuple=True)[0]
    if len(indices_to_remove) > 0:
        logger.info(f"number of pruned points: {len(indices_to_remove)}\n")

    persistent_mask[indices_to_remove] = False
    W.data[~persistent_mask] = 0.0

    '''
    indices_to_remove_sigmax = (torch.sigmoid(W[:, 0]) < 0.01)
    indices_to_remove_sigmay = (torch.sigmoid(W[:, 1]) < 0.01)
    indices_to_remove_sigmaxy = (indices_to_remove_sigmax & indices_to_remove_sigmay).nonzero(as_tuple=True)[0]

    if len(indices_to_remove_sigmaxy) > 0:
        logger.info(f"number of pruned points with lower sigmaxy: {len(indices_to_remove_sigmaxy)}\n")

    persistent_mask[indices_to_remove_sigmaxy] = False
    W.data[~persistent_mask] = 0.0
    '''
    return W, persistent_mask


if __name__ == "__main__":
    signmax = torch.rand([6, 2])
    coords = torch.rand([6, 2])
    print(signmax)
    print(coords)
    get_points_inlcude_overlap_in_patches(coords=coords, signmax=signmax, h=1, w=1, hn=3, wn=4)