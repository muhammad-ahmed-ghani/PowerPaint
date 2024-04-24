import os
import sys
import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers.utils import load_image
from diffusers import DPMSolverMultistepScheduler

from powerpaint_v2.BrushNet_CA import BrushNetModel
from powerpaint_v2.pipeline_PowerPaint_Brushnet_CA import (
    StableDiffusionPowerPaintBrushNetPipeline,
)
from powerpaint_v2.power_paint_tokenizer import PowerPaintTokenizer
from powerpaint_v2.unet_2d_condition import UNet2DConditionModel

torch.set_grad_enabled(False)
root_model_dir = os.environ.get("MODEL_DIR", None)
base_model_path = os.path.join(root_model_dir, "realisticVisionV60B1_v51VAE/")
torch_dtype = torch.bfloat16

global pipe

unet = UNet2DConditionModel.from_pretrained(
    base_model_path,
    subfolder="unet",
    torch_dtype=torch_dtype,
)
brushnet = BrushNetModel.from_pretrained(
    os.path.join(root_model_dir, "PowerPaint_Brushnet/"),
    torch_dtype=torch_dtype,
)
text_encoder_brushnet = CLIPTextModel.from_pretrained(
    os.path.join(root_model_dir, "text_encoder_brushnet/"),
    torch_dtype=torch_dtype,
)

pipe = StableDiffusionPowerPaintBrushNetPipeline.from_pretrained(
    base_model_path, brushnet=brushnet,text_encoder_brushnet = text_encoder_brushnet, torch_dtype=torch_dtype, low_cpu_mem_usage=False, safety_checker=None, unet=unet
)

pipe.tokenizer = PowerPaintTokenizer(CLIPTokenizer.from_pretrained(os.path.join(root_model_dir,"tokenizer/")))
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

def add_task(control_type):
    # print(control_type)
    if control_type == 'object-removal':
        promptA = 'P_ctxt'
        promptB = 'P_ctxt'
        negative_promptA = 'P_obj'
        negative_promptB = 'P_obj'
    elif control_type == 'context-aware':
        promptA = 'P_ctxt'
        promptB = 'P_ctxt'
        negative_promptA = ''
        negative_promptB = ''
    elif control_type == 'shape-guided':
        promptA = 'P_shape'
        promptB = 'P_ctxt'
        negative_promptA = 'P_shape'
        negative_promptB = 'P_ctxt'
    elif control_type == 'image-outpainting':
        promptA = 'P_ctxt'
        promptB = 'P_ctxt'
        negative_promptA = 'P_obj'
        negative_promptB = 'P_obj'
    else:
        promptA = 'P_obj'
        promptB = 'P_obj'
        negative_promptA =  'P_obj'
        negative_promptB =  'P_obj'

    return promptA, promptB, negative_promptA, negative_promptB


@torch.inference_mode()
def predict(input_image, prompt, fitting_degree, ddim_steps, scale, seed,
            negative_prompt, task):
    # size1, size2 = input_image['image'].convert('RGB').size

    # if task!='image-outpainting':
    #     if size1 < size2:
    #         input_image['image'] = input_image['image'].convert('RGB').resize(
    #             (640, int(size2 / size1 * 640)))
    #     else:
    #         input_image['image'] = input_image['image'].convert('RGB').resize(
    #             (int(size1 / size2 * 640), 640))
    # else:
    #     if size1 < size2:
    #         input_image['image'] = input_image['image'].convert('RGB').resize(
    #             (512, int(size2 / size1 * 512)))
    #     else:
    #         input_image['image'] = input_image['image'].convert('RGB').resize(
    #             (int(size1 / size2 * 512), 512))

    if task=='image-outpainting' or task == 'context-aware':
        prompt = prompt + ' empty scene'
    if task=='object-removal':
        prompt = prompt + ' empty scene blur'
        
    # if vertical_expansion_ratio!=None and horizontal_expansion_ratio!=None:
    #     o_W,o_H = input_image['image'].convert('RGB').size
    #     c_W = int(horizontal_expansion_ratio*o_W)
    #     c_H = int(vertical_expansion_ratio*o_H)

    #     expand_img = np.ones((c_H, c_W,3), dtype=np.uint8)*127
    #     original_img = np.array(input_image['image'])
    #     expand_img[int((c_H-o_H)/2.0):int((c_H-o_H)/2.0)+o_H,int((c_W-o_W)/2.0):int((c_W-o_W)/2.0)+o_W,:] = original_img

    #     blurry_gap = 10

    #     expand_mask = np.ones((c_H, c_W,3), dtype=np.uint8)*255
    #     if vertical_expansion_ratio == 1 and horizontal_expansion_ratio!=1:
    #         expand_mask[int((c_H-o_H)/2.0):int((c_H-o_H)/2.0)+o_H,int((c_W-o_W)/2.0)+blurry_gap:int((c_W-o_W)/2.0)+o_W-blurry_gap,:] = 0
    #     elif vertical_expansion_ratio != 1 and horizontal_expansion_ratio!=1:
    #         expand_mask[int((c_H-o_H)/2.0)+blurry_gap:int((c_H-o_H)/2.0)+o_H-blurry_gap,int((c_W-o_W)/2.0)+blurry_gap:int((c_W-o_W)/2.0)+o_W-blurry_gap,:] = 0
    #     elif vertical_expansion_ratio != 1 and horizontal_expansion_ratio==1:
    #         expand_mask[int((c_H-o_H)/2.0)+blurry_gap:int((c_H-o_H)/2.0)+o_H-blurry_gap,int((c_W-o_W)/2.0):int((c_W-o_W)/2.0)+o_W,:] = 0
        
    #     input_image['image'] = Image.fromarray(expand_img)
    #     input_image['mask'] = Image.fromarray(expand_mask)

        

    promptA, promptB, negative_promptA, negative_promptB = add_task(task)
    img = np.array(input_image['image'].convert('RGB'))

    W = int(np.shape(img)[0] - np.shape(img)[0] % 8)
    H = int(np.shape(img)[1] - np.shape(img)[1] % 8)
    input_image['image'] = input_image['image'].resize((H, W))
    input_image['mask'] = input_image['mask'].resize((H, W))

    np_inpimg = np.array(input_image['image'])
    np_inmask = np.array(input_image['mask'])/255.0

    np_inpimg = np_inpimg*(1-np_inmask)

    input_image['image'] = Image.fromarray(np_inpimg.astype(np.uint8)).convert("RGB")

    global pipe
    result = pipe(
            promptA = promptA, 
            promptB = promptB,
            promptU = prompt,
            tradoff=fitting_degree,
            tradoff_nag=fitting_degree,
            image = input_image['image'].convert('RGB'), 
            mask = input_image['mask'].convert('RGB'), 
            num_inference_steps=ddim_steps, 
            generator=torch.Generator("cuda").manual_seed(seed),
            brushnet_conditioning_scale=1.0,
            negative_promptA = negative_promptA,
            negative_promptB = negative_promptB,
            negative_promptU = negative_prompt,
            guidance_scale = scale,
            width=H,
            height=W,
        ).images[0]
    # mask_np = np.array(input_image['mask'].convert('RGB'))
    # red = np.array(result).astype('float') * 1
    # red[:, :, 0] = 180.0
    # red[:, :, 2] = 0
    # red[:, :, 1] = 0
    # result_m = np.array(result)
    # result_m = Image.fromarray(
    #     (result_m.astype('float') * (1 - mask_np.astype('float') / 512.0) +
    #      mask_np.astype('float') / 512.0 * red).astype('uint8'))
    # m_img = input_image['mask'].convert('RGB').filter(
    #     ImageFilter.GaussianBlur(radius=3))
    # m_img = np.asarray(m_img) / 255.0
    # img_np = np.asarray(input_image['image'].convert('RGB')) / 255.0
    # ours_np = np.asarray(result) / 255.0
    # ours_np = ours_np * m_img + (1 - m_img) * img_np
    # result_paste = Image.fromarray(np.uint8(ours_np * 255))

    # dict_res = [input_image['mask'].convert('RGB'), result_m]

    # dict_out = [result]

    return result



def infer(input_image, text_guided_prompt=None, text_guided_negative_prompt=None,
          shape_guided_prompt=None, shape_guided_negative_prompt=None, fitting_degree=1.0,
          ddim_steps=50, scale=12, seed=None, task='text-guided', vertical_expansion_ratio=None,
          horizontal_expansion_ratio=None, outpaint_prompt=None, outpaint_negative_prompt=None,
          removal_prompt=None, removal_negative_prompt=None, context_prompt="",
          context_negative_prompt=""):
    
    if task == 'text-guided':
        prompt = text_guided_prompt
        negative_prompt = text_guided_negative_prompt
    elif task == 'shape-guided':
        prompt = shape_guided_prompt
        negative_prompt = shape_guided_negative_prompt
    elif task == 'object-removal':
        prompt = removal_prompt
        negative_prompt = removal_negative_prompt
    elif task == 'context-aware':
        prompt = context_prompt
        negative_prompt = context_negative_prompt
    elif task == 'image-outpainting':
        prompt = outpaint_prompt
        negative_prompt = outpaint_negative_prompt
        return predict(input_image, prompt, fitting_degree, ddim_steps, scale,
                       seed, negative_prompt, task)
    else:
        task = 'text-guided'
        prompt = text_guided_prompt
        negative_prompt = text_guided_negative_prompt

    return predict(input_image, prompt, fitting_degree, ddim_steps, scale,
                       seed, negative_prompt, task)
