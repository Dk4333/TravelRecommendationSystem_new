import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import time
import os
import torch
from sentence_transformers import SentenceTransformer, util
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from pywebio.input import *
from pywebio.output import *
from pywebio.platform.flask import webio_view
from pywebio import STATIC_PATH
from flask import Flask

from transformers import pipeline

# ------------------ Configuration ------------------

# Set Hugging Face token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "travelchatbot"

# Load Hugging Face chatbot
# Load Hugging Face chatbot with a better model
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, logging

# Silence warnings if no internet
logging.set_verbosity_error()

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")


# Create the chatbot pipeline
chatbot = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    config={"max_length": 100, "do_sample": False, "num_return_sequences": 1}
)
chat_history = [] # to keep the chat history 

# Flask app
app = Flask(__name__)
BASE_IMAGE_PATH = os.path.join(os.getcwd(), 'DestinationPics')



# ------------------ Load Data ------------------

df = pd.read_csv('travel_destinations.csv')
cities = list(df['City'])
description = list(df['description'])

index_destination_dict = {i: df.loc[i]['City'] for i in range(len(df))}
destination_index_dict = {df.loc[i]['City']: i for i in range(len(df))}

df1 = pd.read_csv('destinations_with_processed_text.csv')
corpus = df1['processed_text'].tolist()
  

# Load SBERT model
from sentence_transformers import SentenceTransformer

model_name = 'all-MiniLM-L6-v2'
sbert_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")




# Prepare embeddings
embedding_cache_path = 'corpus_embeddings.pt'

corpus = df1['processed_text'].tolist()

if os.path.exists(embedding_cache_path):
    corpus_embeddings = torch.load(embedding_cache_path)
else:
    corpus_embeddings = sbert_model.encode(corpus, convert_to_tensor=True)
    torch.save(corpus_embeddings, embedding_cache_path)





# ------------------ Core Functions ------------------

def previously_visited_destination(prev_city, threshold=0.0):
    # Get the index of the current city
    idx = destination_index_dict[str(prev_city)]

    # Get the SBERT embedding of the current (visited) city
    prev_city_embedding = corpus_embeddings[idx]

    # Compute cosine similarity with all other cities
    cosine_scores = util.pytorch_cos_sim(prev_city_embedding, corpus_embeddings)[0]

    # Exclude the current city from the similarity list
    similarity_list = [
        (score.item(), i) for i, score in enumerate(cosine_scores)
        if i != idx and score.item() >= threshold
    ]

    # Sort in descending order of similarity
    return sorted(similarity_list, reverse=True)




def free_text_based_query():
    from sentence_transformers import SentenceTransformer, util  # Add at top if not already

    free_text = textarea('Enter a free text', rows=3, placeholder='e.g. snow winter trekking, lake boating tiger...')
    number_of_recommendations = input("Enter the number of recommendations", type=NUMBER)

    # Generate embedding for the input query
    query_embedding = sbert_model.encode(free_text, convert_to_tensor=True)

    # Compute cosine similarity between the query and all descriptions
    cosine_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]

    # Get top-k most similar destinations
    top_results = torch.topk(cosine_scores, k=number_of_recommendations)

    final_rec = [index_destination_dict[int(idx)] for idx in top_results[1]]

    for city in final_rec:
        matched_rows = df[df['City'].str.strip().str.lower() == city.strip().lower()]

        if not matched_rows.empty:
            price = matched_rows['Avg Expense Per Day'].values[0]
            link = matched_rows['link 1'].values[0]
        else:
            price = "N/A"
            link = "#"

        put_html('<hr>')
        put_markdown(f"### 🌆 *{city}*")
        put_text(f"💰 Avg Expense per Day: ₹{price}")
        put_markdown(f"[🔗 More Info]({link})")

        pic_path = os.path.join(BASE_IMAGE_PATH, f"{city}.jpg")
        try:
            img = open(pic_path, 'rb').read()
            put_image(img, width='100%')
        except:
            put_text(f"(Image not found: {pic_path})")




chat_history = []  # Global list to hold chat messages

def chat_with_ai():
    global chat_history

    put_markdown("### 💬 Ask anything about travel!")

    # Show chat history
    if chat_history:
        put_markdown("#### 🕘 Previous Conversation")
        for line in chat_history[-5:]:
            put_text(line)

    # Options for user
    option = actions(label="🔧 Options", buttons=["Continue Chat", "Clear Chat History"])
    if option == "Clear Chat History":
        chat_history = []
        put_markdown("✅ Chat history cleared. Start a new conversation!")
        return  # Exit function early

    # Get user input
    user_question = input("Your question:", type=TEXT)
    chat_history.append(f"User: {user_question}")

    # Encode new input
    new_input = tokenizer.encode(user_question + tokenizer.eos_token, return_tensors='pt')

    try:
        # Use chat history if it exists
        if len(chat_history) > 1:
            history_texts = [line for line in chat_history if isinstance(line, str)]
            history_ids = [tokenizer.encode(line + tokenizer.eos_token, return_tensors='pt') for line in history_texts]
            bot_input_ids = torch.cat(history_ids + [new_input], dim=-1)
        else:
            bot_input_ids = new_input

        # Generate response
        output_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
        output_text = tokenizer.decode(output_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

        chat_history.append(f"Bot: {output_text.strip()}")
        put_markdown(f"**Chatbot Answer:**\n{output_text.strip()}")

    except Exception as e:
        put_text(f"(HuggingFace Error: {str(e)})")




def select_recommendation_system():
    option = select('Which type of recommendation system would you prefer?', [
        'Recommendation based on free text-based query',
        'Recommendations similar to previously visited destination'
    ])
    if option == 'Recommendation based on free text-based query':
        free_text_based_query()
    else:
        prev_city = select('Select the previously visited travel destination', cities)
        recommendations_list = previously_visited_destination(prev_city)
        number_of_recommendations = input("Enter the number of recommendations", type=NUMBER)
        for element in recommendations_list[:number_of_recommendations]:
            city = index_destination_dict[element[1]]

            matched_rows = df[df['City'].str.strip().str.lower() == city.strip().lower()]

            if not matched_rows.empty:
             price = matched_rows['Avg Expense Per Day'].values[0]
             link = matched_rows['link 1'].values[0]
            else:
             price = "N/A"
             link = "#"

            put_html('<hr>')
            put_markdown(f"### 🌆 *{city}*")
            put_text(f"💰 Avg Expense per Day: ₹{price}")
            put_markdown(f"[🔗 More Info]({link})")

            pic_path = os.path.join(BASE_IMAGE_PATH, f"{city}.jpg")
            try:
                img = open(pic_path, 'rb').read()
                put_image(img, width='100%')
            except:
                put_text(f"(Image not found: {pic_path})")




def explore():
    put_markdown('## Please wait! Your request is being processed!')
    put_processbar('bar')
    for i in range(1, 11):
        set_processbar('bar', i / 10)
        time.sleep(0.1)

    for i in range(len(df)):
        city = cities[i]

        # Safely get Holidify link
        matched_row = df[df['City'].str.strip().str.lower() == city.strip().lower()]
        if not matched_row.empty:
            link = matched_row['link 1'].values[0]
        else:
            link = "#"

        put_html('<hr>')
        put_markdown(f"# *`{city}`*")
        put_markdown(f"[🔗 Explore more ]({link})")

        pic_path = os.path.join(BASE_IMAGE_PATH, f"{city}.jpg")
        try:
            img = open(pic_path, 'rb').read()
            put_image(img, width='100%')
        except:
            put_text(f"(Image not found: {pic_path})")

    put_markdown("# *In case of copyright issues, please drop an email to `mayanknarwal0506@gmail.com`*")

    try:
        img = open(os.path.join(BASE_IMAGE_PATH, 'India_1.jpg'), 'rb').read()
        put_image(img, width='1500px')
    except:
        put_text("(Image not found: India_1.jpg)")



def choices():
    try:
        logo_path = os.path.join(BASE_IMAGE_PATH, 'logo.jpg')
        img = open(logo_path, 'rb').read()
        put_image(img, width='700px')
    except FileNotFoundError:
        put_text(f"(Image not found: {logo_path})")

    #put_markdown('# **Welcome to Wayfind.AI**')
    answer = radio("Choose one", options=[
        'Explore Incredible India!',
        'Get Travel Recommendations',
        'Chat with AI Assistant'
    ])

    if answer == 'Explore Incredible India!':
        explore()
    elif answer == 'Get Travel Recommendations':
        put_text("Let's get started!")
        select_recommendation_system()
    elif answer == 'Chat with AI Assistant':
        chat_with_ai()


# ------------------ Run App ------------------

app.add_url_rule('/', 'webio_view', webio_view(choices), methods=['GET', 'POST', 'OPTIONS'])

if __name__ == '__main__':
    app.run(host='localhost', port=8080)

