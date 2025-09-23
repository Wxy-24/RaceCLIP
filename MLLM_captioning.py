
import requests
from PIL import Image
import numpy as np
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from datetime import datetime
import pandas as pd
import os
from torch.utils.data import Dataset
from tqdm import tqdm
import pickle

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria, process_images
import math



class capDataset(Dataset):
    def __init__(self, csv_file, root_dir):
        self.clr_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
    def __len__(self):
        return len(self.clr_frame)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        name=self.clr_frame.iloc[idx, 0]
        text = self.clr_frame.iloc[idx, 1]
        sample = {'image': name, 'text': text}
        sample = sample['image'], sample['text']
        return sample


with open('/lustre/fswork/projects/rech/dvj/uyk23wk/CUI2def_NewROCO.pkl', 'rb') as f:
    cui2def=pickle.load(f)
with open('/lustre/fswork/projects/rech/dvj/uyk23wk/img2rel_NewROCO.pkl', 'rb') as f:
    img2rel=pickle.load(f)

# model_id = "/home/wxy/下载/llava-1.5-7b-hf-bnb-4bit"
# model = LlavaForConditionalGeneration.from_pretrained(
#     model_id, 
#     torch_dtype=torch.float16, 
#     low_cpu_mem_usage=True, 
#     load_in_4bit=True
# ).to(0)
# processor = AutoProcessor.from_pretrained(model_id)
# conversation = [
#     {

#       "role": "user",
#       "content": [
#           {"type": "text", "text": "provide a clinical description of this medical image"},
#           {"type": "image"},
#         ],
#     },
# ]
# prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
# raw_image = Image.open('/home/wxy/work/ROCO/data/test/radiology/images/ROCO_00016.jpg')
# too_image = Image.open('/home/wxy/work/ROCO/data/test/radiology/images/ROCO_00025.jpg')
# inputs = processor(images=[raw_image,too_image], text=[prompt,prompt], return_tensors='pt').to(0, torch.float16)
# output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
# # cap=processor.decode(output[0][2:], skip_special_tokens=True).split(':')[-1]
# cap=processor.batch_decode(output, skip_special_tokens=True)
# res=[i.split(':')[-1] for i in cap]
# for r in res:
#     print(r)


model_id = "/lustre/fswork/projects/rech/dvj/uyk23wk/xiaoyang/llava-med-v1.5-mistral-7b"
tokenizer, model, image_processor, context_len = load_pretrained_model(model_id, None, model_id.split('/')[-1])

start=datetime.now()
# for split in ['valid']:


# ########## batch inference ########

# for split in ['train','valid']:
#     rewrite=[]
#     csv=f'/lustre/fswork/projects/rech/dvj/uyk23wk/xiaoyang/ConVIRT/NEW-ROCO-{split}.csv'
#     root=f'/lustre/fsn1/projects/rech/dvj/uyk23wk/xiaoyang/ImageCLEFmedical_Caption_2023_{split}_images/{split}/'
#     val_dataset = capDataset(csv_file=csv,root_dir=root)
#     val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=16,num_workers=12)
#     df=pd.read_csv(csv)
#     img_list=df.iloc[:,0].tolist()
#     cap_list=df.iloc[:,1].tolist()
#     for step, (x_v, x_u) in enumerate(val_loader):
#         bs=len(x_u)

#         qs = "provide a clinical description of this medical image"
#         cur_prompt = qs
#         if model.config.mm_use_im_start_end:
#             qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
#         else:
#             qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
#         conv = conv_templates["mistral_instruct"].copy()
#         conv.append_message(conv.roles[0], qs)
#         conv.append_message(conv.roles[1], None)
#         prompt = conv.get_prompt()
#         input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
#         x_v=[Image.open(root+i) for i in x_v]
#         image_tensor = process_images(x_v, image_processor, model.config)[0]
#         begin=datetime.now()
#         z=torch.cat([input_ids]*bs,dim=0)
#         print(z.shape)
#         with torch.inference_mode():
#             output_ids = model.generate(
#                 torch.cat([input_ids]*bs,dim=0),
#                 images=image_tensor.unsqueeze(0).half().cuda(),
#                 do_sample= False,
#                 temperature=0.7,
#                 top_p=1.0,
#                 num_beams=1,
#                 max_new_tokens=256,
#                 use_cache=True)
#         response = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
#         for i in response:
#             rewrite.append(i)

#         end=datetime.now()

#         print(f'batch {step}: GENERATION time cost: {end-begin}')




########## single inference ########
for split in ['valid','train']:
    rewrite,instruction=[],[]
    csv=f'/lustre/fswork/projects/rech/dvj/uyk23wk/xiaoyang/ConVIRT/NEW-ROCO-{split}.csv'
    root=f'/lustre/fsn1/projects/rech/dvj/uyk23wk/xiaoyang/ImageCLEFmedical_Caption_2023_{split}_images/{split}/'
    df=pd.read_csv(csv)
    img_list=df.iloc[:,0].tolist()
    cap_list=df.iloc[:,1].tolist()
    cui_list=df.iloc[:,2].tolist()

    for idx,q in enumerate(cap_list):
        begin=datetime.now()

        ##########CUI+DEF################
        cuis=cui_list[idx][2:-2].split("', '")
        qs="This medical image is associated with following medical concepts:\n"
        for num,cui in enumerate(cuis):
            qs=qs+f'{num+1}. {cui2def[cui]}\n'
        # qs=qs + 'Please provide a clinical description of this medical image based on given prompts.'
        # qs=qs + 'Provide a radiology-style report for this image including clinical findings based on given prompts.'
        # qs=qs + 'What can be observed in this medical image? List any abnormalities and specify the anatomical region based on given prompts.'
        qs=qs + 'Generate a structured description of the image, covering: (1) anatomical structures visible, (2) abnormal findings, and (3) clinical significance based on given prompts.'

        ##########CUI+REL################
        # cuis=cui_list[idx][2:-2].split("', '")
        # qs="This medical image is associated with following medical concepts:\n"
        # for num,cui in enumerate(cuis):
        #     entity=cui2def[cui].split(':')[0]
        #     qs=qs+f'{num+1}.{entity},'
        # for num,cui in enumerate(cuis):
        #     qs=qs+f'{num+1}. {cui2def[cui]}\n'
        # total_rel=img2rel[img_list[idx]]
        # qs=qs+"And here are the relationships between the asscociated concepts with other related concepts: e.g.,"
        # for rel in total_rel:
        #     qs=qs+f'{rel},'
        # qs=qs + '\nPlease provide a clinical description of this medical image based on given prompts.'


        # qs = "provide a clinical description of this medical image"
        # qs = "Enrich the following description of this medical images based on clinical observations:"+ q
        cur_prompt = qs
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates["mistral_instruct"].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        print(type(prompt),prompt)
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        print('input_ids.shape: ',input_ids.shape)
        image = Image.open(root+img_list[idx])
        image_tensor = process_images([image], image_processor, model.config)[0]

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                do_sample= False,
                temperature=0.7,
                top_p=1.0,
                num_beams=1,
                max_new_tokens=512,
                use_cache=True)
        response = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        end=datetime.now()
        rewrite.append(response)
        instruction.append(qs)
        print(f'{img_list[idx]}: GENERATION time cost: {end-begin}')


    df_re = pd.DataFrame({'A':img_list,'B':rewrite,'C':instruction})
    df_re.to_csv(f'/lustre/fswork/projects/rech/dvj/uyk23wk/ConVIRT/rewrite/NEW-ROCO-{split}_caption_CUI2DEF-llava-med_prompt4.csv',index=False)

finish=datetime.now()
print(f'TOTAL RECAPTION time cost: {finish-start}')





