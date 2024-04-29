


import torch
from utils import *
import torch.nn.parallel

def generate_2D_gaussian_splatting(image_height, image_width, patch_h_number, patch_w_number, sigma_values, rotation_angles, rgbas, positions, up_ratio=1, device="cuda"):
    
    eps = 0.00001
    image_height = image_height * up_ratio
    image_width = image_width * up_ratio
    
    coords_dict = get_points_in_patches(coords=positions, h=1, w=1,  hn=patch_h_number, wn=patch_w_number)
    results = torch.zeros([image_height, image_width, 3]).to(device)
    
    xs = torch.linspace(0, 1, steps=image_width)
    ys = torch.linspace(0, 1, steps=image_height)
    xx, yy = torch.meshgrid(xs, ys, indexing='xy')
    
    patch_h = image_height//patch_h_number
    patch_w = image_width//patch_w_number
    patches_tensor = torch.zeros([patch_h_number, patch_w_number, patch_h, patch_w, 2]).to(device)
    

    for i in range(patch_h_number):
        for j in range(patch_w_number):
            # Calculate the boundaries of the current patch
            patch_start_x = int(j * patch_w)
            patch_end_x = int((j + 1) * patch_w)
            patch_start_y = int(i * patch_h)
            patch_end_y = int((i + 1) * patch_h)
            patches_tensor[i, j, :, :, :] = torch.cat([yy[patch_start_y:patch_end_y, patch_start_x:patch_end_x].unsqueeze(-1), xx[patch_start_y:patch_end_y, patch_start_x:patch_end_x].unsqueeze(-1)], dim=-1)
    
    
    for ii in range(patch_h_number):
        for jj in range(patch_w_number):
            bn = coords_dict[f'{ii}_{jj}']
            
            try:
                sbn = len(bn)
            except:
                if isinstance(bn, int):
                    bn = [bn]
                    sbn = 1

            if sbn > 0:
                grid_x = patches_tensor[ii, jj, :, :, :].view(-1, 1, 2, 1)
                b_colors = rgbas[bn]
                #print(bn, '----', rotation_angles.shape, '--')
                b_scales   = sigma_values[bn]
                b_rotation = rotation_angles[bn]
                #print(positions.shape, bn, up_ratio, )
                b_coord    = positions[bn] * up_ratio
                x = grid_x - b_coord.unsqueeze(-1)
                

                zero_mask = b_scales == 0
                # Replace zero values with 0.001
                #b_scales[zero_mask] = 0.001
                #b_scales = b_scales#torch.clamp(b_scales, min=0.001)
                #b_rotation = torch.clamp(b_rotation, min=0.0001, max=0.999)
                b_scales = b_scales * up_ratio

                scale_matrices = torch.diag_embed(b_scales)
                cosines = torch.cos(b_rotation * torch.pi)
                sines = torch.sin(b_rotation * torch.pi)
                
                #sines = torch.clamp(sines, min=0.001, max=0.99)
                #cosines = torch.clamp(cosines, min=0.001, max=0.99)
                rot_matrices = torch.cat([cosines, -sines, sines, cosines], 1).reshape(-1, 2, 2)
                covariance_matrices = (rot_matrices @ scale_matrices @ torch.transpose(scale_matrices, -2, -1) @ torch.transpose(rot_matrices, -2, -1).to(device))
                
                linalg_solve = torch.linalg.solve(covariance_matrices, x)

                gaussians = torch.exp(-0.5 * torch.transpose(x, -2, -1) @ linalg_solve)
                #print(scale_matrices.shape, gaussians[3039,153])
                
                # "normalize" individual Gaussians to max 1
                norm_gaussians = gaussians / (gaussians.max() + eps)

                #nan_mask = torch.isnan(norm_gaussians)
                #print(nan_mask.sum())
                # Replace NaN values with 1
                #norm_gaussians[nan_mask] = 0.5


                #check_for_nan({"results": results, "b_scales":b_scales,"scale_matrices": scale_matrices, "norm_gaussians": norm_gaussians, "rot_matrices":rot_matrices, "x":x, "covariance_matrices":covariance_matrices, "linalg_solve":linalg_solve, "gaussians":gaussians})
                
                #if torch.isnan(norm_gaussians).any():
                #    raise("Tensor contains NaN values.")

              
                # calculate alpha_i (cf. Eq. 3 in 3DGS paper)
                alpha_is = norm_gaussians.squeeze(-1).squeeze(-1)* b_colors[:, 3]
                product = torch.cumprod(torch.cat([torch.ones((alpha_is.shape[0], 1), device=device), (1 - alpha_is)[..., :-1]], -1), 1).unsqueeze(-1).to(device)
                rgb = b_colors[:, :3] * alpha_is.unsqueeze(-1) * product
                rgb = rgb.sum(1).view(patch_h, patch_w, 3)
                
                results[int(ii*patch_h): int((ii+1)*patch_h), int(jj*patch_w): int((jj+1)*patch_w), :] = rgb
                
                #check_for_nan({"results": results, "norm_gaussians": norm_gaussians, "rot_matrices":rot_matrices, "x":x, "covariance_matrices":covariance_matrices, "linalg_solve":linalg_solve, "gaussians":gaussians})
                
            else:
                print(ii, jj ,'---')
    return results









if __name__ == "__main__":
    image_height = 512
    image_width = 512
    patch_h_number = 2
    patch_w_number = 2
    device = "cuda"
    sigma_values = torch.rand([100, 2]).to(device)
    rotation_angles = torch.rand([100, 1]).to(device)
    rgbas = torch.rand([100, 4]).to(device)
    positions  = torch.rand([100, 2]).to(device)
    
    generate_2D_gaussian_splatting(image_height, image_width, patch_h_number, patch_w_number, sigma_values, rotation_angles, rgbas, positions, device="cuda")
