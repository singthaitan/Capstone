import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import faiss
import numpy as np

# Load pre-trained model and tokenizer
model_name = "t5-small"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load job listings data
job_listings = [{"description": "Job listing 1"}, {"description": "Job listing 2"}, {"description": "Job listing 3"}]  # load your job listings data here

# Load sentence transformer model
ST = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")

# Embed job listings using sentence transformer model
embedded_job_listings = [{"embeddings": ST.encode(description["description"])} for description in job_listings]

# Create a FAISS index
index = faiss.IndexFlatL2(1024)  # 768 is the embedding dimension
embeddings = [embedding["embeddings"] for embedding in embedded_job_listings]
embeddings_array = np.array(embeddings)
print(embeddings_array.shape)
index.add(np.array(embeddings))


# Define a search function
def search(query: str, k: int = 3):
    embedded_query = ST.encode(query)
    scores, retrieved_examples = index.search(np.array([embedded_query]), k=k)
    return scores, retrieved_examples

# Create a Streamlit app
st.set_page_config(page_title="Job Search Chatbot")

# Create a sidebar for user input and settings
with st.sidebar:
    st.title("Job Search Chatbot")
    user_input = st.text_input("Enter your query:")
    temperature = st.slider("Temperature", min_value=0.01, max_value=5.0, value=0.1, step=0.01)
    top_p = st.slider("Top P", min_value=0.01, max_value=1.0, value=0.9, step=0.01)
    max_length = st.slider("Max Length", min_value=32, max_value=128, value=120, step=8)

# Create a chat window
chat_window = st.container()

# Define a function to update the chat window
def update_chat_window(user_input):
    scores, retrieved_examples = search(user_input)
    recommended_jobs = []
    for i, score in enumerate(scores[0]):
        recommended_jobs.append(job_listings[i]["description"])
    with chat_window:
        st.write("You: " + user_input)
        st.write("Recommended Jobs:")
        for job in recommended_jobs:
            st.write(job)
        # Generate a response based on the user's query
        response = generate_response(user_input, recommended_jobs)
        st.write("Chatbot: " + response)

# Define a function to generate a response
def generate_response(user_input, recommended_jobs):
    # Use the T5 model to generate a response
    input_ids = tokenizer.encode_plus(user_input, 
                                        add_special_tokens=True, 
                                        max_length=512, 
                                        return_attention_mask=True, 
                                        return_tensors='pt')
    output = model.generate(input_ids['input_ids'], 
                             attention_mask=input_ids['attention_mask'], 
                             max_length=128, 
                             temperature=0.1, 
                             top_p=0.9)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

# Create a button to send the user input
if st.button("Send"):
    update_chat_window(user_input)

# Run the app
if __name__ == "__main__":
    st.write("Hello, World!")