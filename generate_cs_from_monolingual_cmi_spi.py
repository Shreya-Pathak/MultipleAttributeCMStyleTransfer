import time
import torch
import os

from transformers import T5Tokenizer
from model import MT5ForStyleConditionalGeneration
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_filepath', type=str, required=True)
parser.add_argument('--output_filepath', type=str, required=True)
args = parser.parse_args()

model_checkpoint="models/translation_multitask_cmi_spi_vec"
input_filepath = args.input_filepath
output_filepath = args.output_filepath

print(f"Reading from : {input_filepath}")
print(f"Writing to : {output_filepath}")

print(f"Model name : {model_checkpoint}")
tokenizer = T5Tokenizer.from_pretrained(model_checkpoint)
model = MT5ForStyleConditionalGeneration.from_pretrained(model_checkpoint, num_attr=2)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device, flush=True)
model.eval()
model.to(device)

batch_size = 64
fw_hi = open(output_filepath, "w")
task_prefix = "to_cm "

def generate(sentences, cmi_scores, spi_scores):
    batch_encoding = tokenizer(sentences, max_length=256, return_tensors="pt", padding=True, truncation=True)
    style_scores = [cmi_scores, spi_scores]
    batch_encoding["input_style_scores"] = torch.transpose(torch.tensor(style_scores, dtype=torch.float32), 0, 1)
    for k,v in batch_encoding.items():
        batch_encoding[k] = v.to(device)
    outputs = model.generate(**batch_encoding, max_length=256)
    return [tokenizer.decode(t, skip_special_tokens=True).strip() for t in outputs]

predictions = []
references = []
with open(input_filepath) as fr:
    i = 0
    batch = []
    batch_cmi = []
    batch_spi = []
    st_time = time.time()
    for line in fr:
        components = line.strip().split('\t')
        if len(components)<3:
            continue
        if len(batch) < batch_size:
            batch.append(task_prefix + " ".join(components[0:-2]))
            batch_cmi.append(float(components[-2]))
            batch_spi.append(float(components[-1]))
        else:
            translated_batch = generate(batch, batch_cmi, batch_spi)
            for en, hi in zip(batch, translated_batch):
                # fw_hi.write(en + "\t" + hi + "\n")
                fw_hi.write(hi + "\n")
                i += 1
                fw_hi.flush()
            batch = [task_prefix + " ".join(components[0:-2])]
            batch_cmi = [float(components[-2])]
            batch_spi = [float(components[-1])]
            if i % (batch_size*100) == 0:
                total_time = time.time() - st_time
                print("Total time for", (batch_size*100), "examples:", total_time)
                print("Time per example:", total_time/(batch_size*100))
                print("Total number of sentences:", i, flush=True)
                st_time = time.time()
    if len(batch) > 0:
        translated_batch = generate(batch, batch_cmi, batch_spi)
        for en, hi in zip(batch, translated_batch):
            # fw_hi.write(en + "\t" + hi + "\n")
            fw_hi.write(hi + "\n")
            i += 1
fw_hi.close()

print("Done")
