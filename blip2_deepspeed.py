# command: deepspeed --num_gpus 1 blip2_deepspeed.py

from PIL import Image
import requests
from transformers import Blip2Processor, Blip2ForConditionalGeneration, Blip2Config
import torch
import deepspeed

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16
)

ds_config = {
    "zero": {
        "stage": 3,
        "overlap_comm": True,
        "contiguous_gradients": True,
        "reduce_bucket_size": 50_000_000,
        "stage3_prefetch_bucket_size": 0.9 * 50_000_000,
        "stage3_param_persistence_threshold": 0,
        "offload_param": {
            "device": "cpu",
            "pin_memory": False
        }
    },
}


ds_engine = deepspeed.init_inference(model, mp_size=1, dtype=torch.float16, replace_with_kernel_inject=True, config=ds_config)

model = ds_engine.module

# model.to(device)
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

prompt = "Question: how many cats are there? Answer:"
inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, torch.float16)


generated_ids = model.generate(**inputs)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
print(generated_text)