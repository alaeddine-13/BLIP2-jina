from jina import Executor, requests
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch

from docarray import BaseDocument, DocumentArray

from docarray.typing import ImageUrl

class ImagePrompt(BaseDocument):
    img: ImageUrl
    prompt: str

class Response(BaseDocument):
    answer: str

class Blip2Executor(Executor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-flan-t5-xl", torch_dtype=torch.float16
        )
        self.model.to('cuda')


    @requests
    def vqa(self, docs: DocumentArray[ImagePrompt], **kwargs) -> DocumentArray[Response]:
        response_docs = DocumentArray[Response]()
        for doc in docs:
            inputs = self.processor(images=doc.img.load(), text=doc.prompt, return_tensors="pt").to('cuda', torch.float16)
            generated_ids = self.model.generate(**inputs)
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            response_docs.append(Response(answer=generated_text))
        return response_docs



from jina import Deployment

with Deployment(uses=Blip2Executor) as dep:
    docs = dep.post(on='/bar', inputs=ImagePrompt(
        img='http://images.cocodataset.org/val2017/000000039769.jpg',
        prompt='Question: how many cats are there ? Answer:'
    ), return_type=DocumentArray[Response])
    print(docs[0].answer)