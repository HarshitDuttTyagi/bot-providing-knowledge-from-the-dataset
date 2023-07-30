# bot-providing-knowledge-from-the-dataset

pandas is a package that allows us to conveniently store and manipulate data in a data structure known as a Dataframe. (This is similar to a Dataframe in R, for those familiar with R.) It’s a very common tool for anyone doing data science in python.
sklearn is the package formally called “scikit-learn”, and contains a wide range of statistical and machine learning methods. It’s another very common package for data scientists in python.
numpy is python’s main numeric library, and allows us to do things like work with arrays, matrices, dot products, etc.
json is a package for interacting with json files. Our data is formatted as a single json file, so this is useful for us here.
os helps us with file management and command-line commands.
openai is a package containing functions that allow us to easily make API calls to OpenAI’s models in python.
Finally, we import cosine_similarity from sklearn, since it’s a specialized function that we need today.

Here's what the model is doing: we have a long piece of text that we want ChatGPT to be able to answer questions about. 
We first break that text up into chunks containing 600 words (technically called “tokens”), where each chunk overlaps 20 words with the following chunk. 
We then send these chunks to OpenAI to obtain their embeddings. When we ask a question about our text, we find the question’s embedding, and use cosine similarity to find the chunk of text that is closest to our question.
We then send a query to ChatGPT that includes our original question, as well as the chunk of text as context.
We loop over all the chunks, and send each one to OpenAI, get back the embedding, and then write a new line to the Dataframe df. 
Note that we are casting the embedding response (a string) to a numpy array. We do this because we will be doing numerical operations on the embedding in just a moment.

Now, let’s define our query and get its embedding. Our query is a simple question: who was the captain of the Excalibur? 
A bit of context: in this episode, a small detail is that one of the crew members was assigned to command a ship for this one episode only, and it’s a minor detail in the plot of the episode. In fact, if you ask ChatGPT this question without giving it the script, it doesn’t know the answer. We’ll see that with the right chunk of text, identified by cosine similarity, ChatGPT can answer correctly.
We calculate the cosine distance from our query to each chunk, and save the chunk that is most similar to a variable called context_chunk.
Finally, we assemble the full query, including the chunk we identified, and send it to ChatGPT via the API:
