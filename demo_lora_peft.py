# Copyright (C) 2022-present Naver Corporation.
import torch
from models.croco import CroCoNet
from PIL import Image
import torchvision.transforms
from torchvision.transforms import ToTensor, Normalize, Compose
import torch.nn.functional as F
from models.criterion import MaskedMSE
from torchmetrics.functional import structural_similarity_index_measure as ssim

from peft import PeftModel

def inference(model, image1, image2, device):

    with torch.inference_mode():
        out, mask, target = model(image1, image2)

    criterion = MaskedMSE(norm_pix_loss=True)
    loss = criterion(out, mask, target)

    # Decode output
    patchified = model.patchify(image1)
    mean = patchified.mean(dim=-1, keepdim=True)
    var = patchified.var(dim=-1, keepdim=True)

    decoded_image = model.unpatchify(out * (var + 1.e-6)**.5 + mean)

    return decoded_image, loss , mask
def main():

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # ----------------------------
    # Image transforms
    # ----------------------------
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    imagenet_mean_tensor = torch.tensor(imagenet_mean).view(1,3,1,1).to(device)
    imagenet_std_tensor = torch.tensor(imagenet_std).view(1,3,1,1).to(device)

    trfs = Compose([
        ToTensor(),
        Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    image1 = trfs(Image.open(r'New_drone_dataset\VisDrone2019-DET-test-dev\images\0000006_00159_d_0000001.jpg').convert('RGB').resize((224,224))).unsqueeze(0).to(device)

    image2 = trfs(Image.open(r'New_drone_dataset\VisDrone2019-DET-test-dev\images\0000006_00611_d_0000002.jpg').convert('RGB').resize((224,224))).unsqueeze(0).to(device)


    # ----------------------------
    # Load base CroCo model
    # ----------------------------
    base_ckpt = torch.load(
        r"pretrained_models\CroCo_V2_ViTBase_SmallDecoder.pth",
        map_location="cpu"
    )

    model = CroCoNet(**base_ckpt.get('croco_kwargs', {})).to(device)
    model.load_state_dict(base_ckpt['model'], strict=True)

    model.eval()


    # ----------------------------
    # Load LoRA adapter (PEFT)
    # ----------------------------

    exp_name = "LoRA_on_ENCODER_Decoder_r16_qkv_proj_peft_filtered_data"
    lora_path = f"output/{exp_name}\checkpoint-{exp_name}-last.pth"
    frame_out_path = f"output/{exp_name}"
    model = PeftModel.from_pretrained(model, lora_path)
    model = model.to(device)

    model.eval()


    decoded_image, loss , mask = inference(model, image1, image2, device)

    decoded_image = decoded_image * imagenet_std_tensor + imagenet_mean_tensor
    input_image = image1 * imagenet_std_tensor + imagenet_mean_tensor
    ref_image = image2 * imagenet_std_tensor + imagenet_mean_tensor

    image_masks = model.unpatchify(
        model.patchify(torch.ones_like(ref_image)) * mask[:,:,None]
    )

    masked_input_image = (1 - image_masks) * input_image


    # ----------------------------
    # Metrics
    # ----------------------------
    mse = F.mse_loss(decoded_image, image1)
    psnr = 10 * torch.log10(1.0 / mse)

    print("MaskedMSE Loss:", loss.item())
    print("Image MSE:", mse.item())
    print("PSNR:", psnr.item(), "dB")

    ssim_value = ssim(decoded_image, image1, data_range=1.0)
    print("SSIM:", ssim_value.item())


    # ----------------------------
    # Visualization
    # ----------------------------
    visualization = torch.cat(
        (ref_image, masked_input_image, decoded_image, input_image),
        dim=3
    )

    B, C, H, W = visualization.shape
    visualization = visualization.permute(1,0,2,3).reshape(C, B*H, W)

    visualization = torchvision.transforms.functional.to_pil_image(
        torch.clamp(visualization, 0, 1)
    )

    fname = f"{frame_out_path}/{exp_name}_{loss.item():.4f}.png"
    visualization.save(fname)

    print("Visualization saved in", fname)





if __name__ == "__main__":
    main()