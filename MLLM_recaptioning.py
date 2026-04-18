import requests
from PIL import Image
import numpy as np
import torch
from datetime import datetime
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, AutoModelForImageTextToText
import pickle


# --- Configuration ---
BATCH_SIZE = 16  # Set your desired batch size here
NUM_WORKERS = 3 # Adjust based on CPU cores available
# ---------------------


with open('/lustre/fswork/projects/rech/dvj/uyk23wk/xiaoyang/CUI2def_NewROCO.pkl', 'rb') as f:
    cui2def=pickle.load(f)


import torch

def print_gpu_memory():
    # Current memory used by tensors
    allocated = torch.cuda.memory_allocated() / 1024**2
    # Total memory reserved by PyTorch
    reserved = torch.cuda.memory_reserved() / 1024**2
    # Peak memory reached
    peak = torch.cuda.max_memory_allocated() / 1024**2
    
    print(f"Allocated: {allocated:.2f} MB | Reserved: {reserved:.2f} MB | Peak: {peak:.2f} MB")



class capDataset(Dataset):
    def __init__(self, csv_file, root_dir):
        self.clr_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        # self.prompt = "Describe this medical image"
        # self.prompt = "Improve the following caption according to your observation of this medical image:"
        self.prompt = "This medical image is associated with following medical concepts:\n"

    def __len__(self):
        return len(self.clr_frame)

    def __getitem__(self, idx):
        filename = self.clr_frame.iloc[idx, 0]
        # We return the filename so we can track it, and the image object
        img_path = os.path.join(self.root_dir, filename)
        
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Create a blank image or handle error as needed to prevent crash
            image = Image.new('RGB', (224, 224), color='black')

        original_text = self.clr_frame.iloc[idx, 1]

        qs=self.prompt
        cuis = self.clr_frame.iloc[idx, 2][2:-2].split("', '")

        ##########CUI+DEF################
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

        
        # We prepare the message structure here, but we don't tokenize yet
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    # {"type": "text", "text": self.prompt+original_text}
                    {"type": "text", "text": qs}
                ]
            }
        ]
        
        return {
            "image": image, 
            "filename": filename, 
            "original_text": original_text, 
            "messages": messages,
            "instruction": qs
        }

# Collate function to handle list of dictionaries from dataset
def custom_collate(batch):
    images = [item['image'] for item in batch]
    filenames = [item['filename'] for item in batch]
    original_texts = [item['original_text'] for item in batch]
    messages = [item['messages'] for item in batch]
    instructions = [item['instruction'] for item in batch]
    return images, filenames, original_texts, messages, instructions


model_id = "/lustre/fswork/projects/rech/dvj/uyk23wk/xiaoyang/medgemma-1.5-4b"

print("Loading model...")
# model = AutoModelForImageTextToText.from_pretrained(
#     model_id,
#     torch_dtype=torch.bfloat16,
#     device_map="auto",
# )

model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="eager"  # <--- Add this line
)

processor = AutoProcessor.from_pretrained(model_id)

# IMPORTANT: For batched generation with decoder models, use left padding
processor.tokenizer.padding_side = "left" 

start = datetime.now()

########## Inference Loop ########
# for split in ['valid', 'train']:
for split in ['train']:
    print(f"Processing split: {split}")
    
    csv_path = f'/lustre/fswork/projects/rech/dvj/uyk23wk/xiaoyang/ConVIRT/NEW-ROCO-{split}2.csv'
    root_path = f'/lustre/fsn1/projects/rech/dvj/uyk23wk/xiaoyang/ImageCLEFmedical_Caption_2023_{split}_images/{split}/'
    
    # Initialize Dataset and DataLoader
    dataset = capDataset(csv_file=csv_path, root_dir=root_path)
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS, 
        collate_fn=custom_collate
    )

    all_rewrites = []
    all_instructions = []
    all_filenames = []
    
    for i, (batch_images, batch_filenames, _, batch_messages, batch_instructions) in enumerate(dataloader):
        batch_start = datetime.now()

        # 1. Convert messages to text prompts (without tokenizing yet)
        # We iterate through the batch of messages to apply the template
        text_prompts = [
            processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in batch_messages
        ]

        # 2. Process the batch (Tokenize text + Process images)
        inputs = processor(
            text=text_prompts,
            images=[[img] for img in batch_images], 
            padding=True,
            return_tensors="pt"
        ).to(model.device, dtype=torch.bfloat16)

        # 3. Generate
        with torch.inference_mode():

            print("Memory Before Inference:")
            print_gpu_memory()
            

            # Get input length for slicing output later
            input_len = inputs["input_ids"].shape[-1] 
            
            generated_ids = model.generate(
                **inputs, 
                max_new_tokens=1000, 
                do_sample=False
            )
            
            # Slice the generated IDs to remove the input prompt
            # Note: generated_ids includes the input tokens
            output_ids = generated_ids[:, input_len:]
            
            print("Memory After Inference:")
            print_gpu_memory()

        # 4. Decode Batch
        decoded_texts = processor.batch_decode(output_ids, skip_special_tokens=True)

        batch_end = datetime.now()
        
        # Store results and print status
        all_rewrites.extend(decoded_texts)
        all_instructions.extend(batch_instructions)
        all_filenames.extend(batch_filenames)

        print(f"Batch {i}/{len(dataloader)} - Size: {len(batch_filenames)} - Time: {batch_end - batch_start}")
        # Optional: Print first generation of batch to check sanity
        # print(f"Sample: {decoded_texts[0]}")

    # Save Results
    df_re = pd.DataFrame({'A': all_filenames, 'B': all_rewrites, 'C': all_instructions})
    save_path = f'/lustre/fswork/projects/rech/dvj/uyk23wk/xiaoyang/ConVIRT/rewrite/NEW-ROCO-{split}2_caption-medGemma1.5-DEF.csv'
    df_re.to_csv(save_path, index=False)
    print(f"Saved results to {save_path}")
    print(f'{split} split RECAPTION time cost: {datetime.now()-start}')

finish = datetime.now()
print(f'TOTAL RECAPTION time cost: {finish-start}')





