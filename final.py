# Q&A Chatbot

from dotenv import load_dotenv
import streamlit as st
import os
import google.generativeai as genai
from PIL import Image

# Load environment variables
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to get Gemini response
def get_gemini_response(input, image):
    model_with_image = genai.GenerativeModel('gemini-pro-vision')
    model_text_only = genai.GenerativeModel('gemini-pro')
    
    if input and image:
        response = model_with_image.generate_content([input, image])
    elif image:
        response = model_with_image.generate_content(image)
    elif input:
        response = model_text_only.generate_content(input)
    else:
        response = "No input provided."
    
    return response.text

# Initialize session state if it doesn't exist
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Function to simulate chatbot response
def chatbot_response(user_input, image):
    return get_gemini_response(user_input, image)

# Initialize Streamlit app
st.set_page_config(page_title="Gemini Image Demo")

# Title of the app
st.title("Gemini Image Chatbot")

# Chat history container
chat_history = st.container()

# Display chat messages
with chat_history:
    for message in st.session_state.messages:
        if message['role'] == 'user':
            st.text_area("User", message['content'], key=message['id'], height=100)
        else:
            st.text_area("Chatbot", message['content'], key=message['id'], height=100)

# User input
input_prompt = st.text_input("Input Prompt: ", key="input")

# Image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
image = ""
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)

# Send button
if st.button("Send"):
    if input_prompt or uploaded_file:
        # Append user's message to session state
        st.session_state.messages.append({"role": "user", "content": input_prompt, "id": len(st.session_state.messages)})

        # Get chatbot response
        response = chatbot_response(input_prompt, image)

        # Append chatbot's response to session state
        st.session_state.messages.append({"role": "chatbot", "content": response, "id": len(st.session_state.messages)})

        # Clear user input
        st.experimental_rerun()
