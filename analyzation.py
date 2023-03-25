import pandas as pd
import numpy as np
import json
import openai
from sklearn.metrics.pairwise import cosine_similarity
import os

# we import the packages we'll need.
# Paste your OpenAI API key here and hit enter
CHUNK_SIZE = 600
OVERLAP = 20
openai.api_key = input("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");
scripts = json.load(open("/content/season1.json", encoding='ascii')) # https://www.kaggle.com/datasets/gjbroughton/start-trek-scripts?resource=download
text = scripts['Game Of Thrones S01E07 You Win Or You Die.srt']['1']
text_list = text.split()
chunks = [text_list[i:i+CHUNK_SIZE] for i in range(0, len(text_list), CHUNK_SIZE-OVERLAP)]
# Here's what the model is doing: we have a long piece of text that we want ChatGPT to be able to answer questions about.
#  We first break that text up into chunks containing 600 words (technically called “tokens”), where each chunk overlaps 20 words with the following chunk. 
#  We then send these chunks to OpenAI to obtain their embeddings. 
#  When we ask a question about our text, we find the question’s embedding, and use cosine similarity to find the chunk of text that is closest to our question.
#   We then send a query to ChatGPT that includes our original question, as well as the chunk of text as context.

# We loop over all the chunks, and send each one to OpenAI, get back the embedding, and then write a new line to the Dataframe df. 
# Note that we are casting the embedding response (a string) to a numpy array. 
# We do this because we will be doing numerical operations on the embedding in just a moment.
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

# Now, let’s define our query and get its embedding. Our query is a simple question: who was the result? 
#  In fact, if you ask ChatGPT this question without giving it the script, it doesn’t know the answer.
#   We’ll see that with the right chunk of text, identified by cosine similarity, ChatGPT can answer correctly.

# We calculate the cosine distance from our query to each chunk, and save the chunk that is most similar to a variable called context_chunk.

# Finally, we assemble the full query, including the chunk we identified, and send it to ChatGPT via the API:


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
# Execute the cell below to find out!
print(response['choices'][0]['text'].strip())
