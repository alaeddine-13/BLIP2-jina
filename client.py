from jina import Client, Document
client = Client(port=12346)

docs = client.post(
        on='/',
        inputs=[Document(
            uri='http://images.cocodataset.org/val2017/000000039769.jpg',
            tags={'prompt': 'Question: Describe the following picture? Answer: '}
        )]
)
print(docs[0].tags['response'])
