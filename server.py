from jina import Executor, Deployment, DocumentArray, requests
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch

class Blip2Executor(Executor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-flan-t5-xl", torch_dtype=torch.float16
        )
        self.model.to('cuda')


    @requests
    def vqa(self, docs: DocumentArray, **kwargs):
        for doc in docs:
            doc.load_uri_to_image_tensor()
            image = doc.tensor
            prompt = doc.tags['prompt']
            inputs = self.processor(images=image, text=prompt, return_tensors="pt").to('cuda', torch.float16)
            generated_ids = self.model.generate(**inputs)
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            doc.tags['response'] = generated_text


with Deployment(uses=Blip2Executor, port=12346, timeout_ready=-1) as dep:
    dep.block()
