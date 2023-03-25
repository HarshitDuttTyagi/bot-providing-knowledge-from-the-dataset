import pandas as pd
import numpy as np
import json
import openai
from sklearn.metrics.pairwise import cosine_similarity
import os

CHUNK_SIZE = 600
OVERLAP = 20
openai.api_key = input("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");
scripts = json.load(open("/content/season1.json", encoding='ascii')) # https://www.kaggle.com/datasets/gjbroughton/start-trek-scripts?resource=download
text = scripts['Game Of Thrones S01E07 You Win Or You Die.srt']['1']
text_list = text.split()
chunks = [text_list[i:i+CHUNK_SIZE] for i in range(0, len(text_list), CHUNK_SIZE-OVERLAP)]
df = pd.DataFrame(columns=['chunk', 'gpt_raw', 'embedding'])
for chunk in chunks:
    f = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=" ".join(chunk),
    )
    df.loc[len(df.index)] = (chunk, f, np.array(f['data'][0]['embedding']))
    df.head()
    query = "what is the result"
f = openai.Embedding.create(
    model="text-embedding-ada-002",
    input=query
)
query_embedding = np.array(f['data'][0]['embedding'])

similarity = []
for arr in df['embedding'].values:
    similarity.extend(cosine_similarity(query_embedding.reshape(1, -1), arr.reshape(1, -1)))
context_chunk = chunks[np.argmax(similarity)]

query_to_send = "CONTEXT: " + " ".join(context_chunk) + "\n\n" + query
response = openai.Completion.create(
  model="text-davinci-003",
  prompt= query_to_send,
  max_tokens=100,
  temperature=0
)
print(query_to_send)
print(response['choices'][0]['text'].strip())
