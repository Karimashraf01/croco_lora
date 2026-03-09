# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

import torch
from models.croco import CroCoNet
from PIL import Image
import torchvision.transforms
from torchvision.transforms import ToTensor, Normalize, Compose



import torch.nn.functional as F
from models.criterion import MaskedMSE
import math


from torchmetrics.functional import structural_similarity_index_measure as ssim


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() and torch.cuda.device_count()>0 else 'cpu')
    r  = 16
    alpha = 32
    dropout = 0.1
    qkv_only = False
    # load 224x224 images and transform them to tensor 
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_mean_tensor = torch.tensor(imagenet_mean).view(1,3,1,1).to(device, non_blocking=True)
    imagenet_std = [0.229, 0.224, 0.225]
    imagenet_std_tensor = torch.tensor(imagenet_std).view(1,3,1,1).to(device, non_blocking=True)
    trfs = Compose([ToTensor(), Normalize(mean=imagenet_mean, std=imagenet_std)])
    # resize to 244x244 to make it divisible by 16 (the patch size)

    image1 = trfs(Image.open(r'New_drone_dataset\VisDrone2019-DET-test-dev\images\0000006_00159_d_0000001.jpg').convert('RGB').resize((224,224))).to(device, non_blocking=True).unsqueeze(0)
    image2 = trfs(Image.open(r'New_drone_dataset\VisDrone2019-DET-test-dev\images\0000006_00611_d_0000002.jpg').convert('RGB').resize((224,224))).to(device, non_blocking=True).unsqueeze(0)
    


    # # load model 
    ckpt = torch.load('pretrained_models/CroCo_V2_ViTBase_SmallDecoder.pth', 'cpu')
    model = CroCoNet( **ckpt.get('croco_kwargs',{})).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    exp_name = "LoRA_r16_qkvproj_encdec"
    ckpt_path = f"output/{exp_name}/checkpoint-{exp_name}-best.pth"
    frame_out_path = f"output/{exp_name}"
    ckpt = torch.load(ckpt_path, map_location="cpu")



    from models.lora import LoRALinear

    for blk in model.enc_blocks:
        blk.attn.qkv = LoRALinear(
            blk.attn.qkv,
            r=r,              # must match training args
            alpha=alpha,
            dropout=dropout,
        )
        if not qkv_only:
            blk.attn.proj = LoRALinear(
                blk.attn.proj,
                r=r,
                alpha=alpha,
                dropout=dropout
            )
    for blk in model.dec_blocks:
        blk.attn.qkv = LoRALinear(
            blk.attn.qkv,
            r=r,              # must match training args
            alpha=alpha,
            dropout=dropout,
        )
        if not qkv_only:
            blk.attn.proj = LoRALinear(
                blk.attn.proj,
                r=r,
                alpha=alpha,
                dropout=dropout
            )


    # Load LoRA weights
    if "lora_state_dict" in ckpt:
        model.load_state_dict(ckpt["lora_state_dict"], strict=False)

    model.eval()

    # forward 
    with torch.inference_mode():
        out, mask, target = model(image1, image2)


    criterion = MaskedMSE(norm_pix_loss=True)
    loss = criterion(out, mask, target)
    

        
    # the output is normalized, thus use the mean/std of the actual image to go back to RGB space 
    patchified = model.patchify(image1)
    mean = patchified.mean(dim=-1, keepdim=True)
    var = patchified.var(dim=-1, keepdim=True)
    decoded_image = model.unpatchify(out * (var + 1.e-6)**.5 + mean)
    # undo imagenet normalization, prepare masked image
    decoded_image = decoded_image * imagenet_std_tensor + imagenet_mean_tensor
    input_image = image1 * imagenet_std_tensor + imagenet_mean_tensor
    ref_image = image2 * imagenet_std_tensor + imagenet_mean_tensor
    image_masks = model.unpatchify(model.patchify(torch.ones_like(ref_image)) * mask[:,:,None])
    masked_input_image = ((1 - image_masks) * input_image)


    # MSE in image space
    mse = F.mse_loss(decoded_image, image1)


    # PSNR
    psnr = 10 * torch.log10(1.0 / mse)

    print("MaskedMSE Loss:", loss.item())
    print("Image MSE:", mse.item())
    print("PSNR:", psnr.item(), "dB")

    
    ssim_value = ssim(decoded_image, image1, data_range=1.0)
    print("SSIM:", ssim_value.item())
    # make visualization
    visualization = torch.cat((ref_image, masked_input_image, decoded_image, input_image), dim=3) # 4*(B, 3, H, W) -> B, 3, H, W*4
    B, C, H, W = visualization.shape
    visualization = visualization.permute(1, 0, 2, 3).reshape(C, B*H, W)
    visualization = torchvision.transforms.functional.to_pil_image(torch.clamp(visualization, 0, 1))
    fname = f"{frame_out_path}/{exp_name}_{loss.item():.4f}.png"
    visualization.save(fname)
    print('Visualization save in '+fname)
    

if __name__=="__main__":
    main()
