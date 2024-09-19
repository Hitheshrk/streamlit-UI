import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


@st.cache_resource
def load_model():
    model_name = "emilyalsentzer/Bio_ClinicalBERT"  # Replace 
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

model, tokenizer = load_model()


def chatbot_response(prompt):
    try:
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(inputs["input_ids"], max_length=500, pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(outputs[:, inputs["input_ids"].shape[-1]:][0], skip_special_tokens=True)
        return response
    except Exception as e:
        return f"An error occurred: {str(e)}"


if 'messages' not in st.session_state:
    st.session_state.messages = []


st.markdown("""
    <style>
        .chat-container {
            background-color: #f9f9f9;
            padding: 10px;
            border-radius: 10px;
            margin-bottom: 10px;
        }
        .chat-user {
            background-color: #e0f7fa;
            padding: 10px;
            border-radius: 10px;
            margin-bottom: 10px;
            color: #00796b;
        }
        .chat-bot {
            background-color: #e8eaf6;
            padding: 10px;
            border-radius: 10px;
            margin-bottom: 10px;
            color: #1a237e;
        }
        .sidebar-text-input .stTextInput > div > div > input {
            font-size: 20px;
        }
    </style>
""", unsafe_allow_html=True)


st.title("Medical Chatbot")


st.markdown("""
**Disclaimer:** This chatbot is for informational purposes only and should not be considered as medical advice. Always consult with a healthcare professional for medical advice and treatment.
""")


with st.sidebar:
    st.markdown('<div class="sidebar-text-input">', unsafe_allow_html=True)
    user_input = st.text_input("I am here to help you, ask me anything: ", "")
    st.markdown('</div>', unsafe_allow_html=True)
    
    if st.button("Send"):
        if user_input:
            
            st.session_state.messages.append({"role": "user", "content": user_input, "id": len(st.session_state.messages)})

           
            conversation_history = "\n".join([f"User: {msg['content']}" if msg['role'] == 'user' else f"Chatbot: {msg['content']}" for msg in st.session_state.messages])
            prompt = f"{conversation_history}\nUser: {user_input}\nChatbot:"

            
            response = chatbot_response(prompt)

            
            st.session_state.messages.append({"role": "chatbot", "content": response, "id": len(st.session_state.messages)})

            
            st.experimental_rerun()


chat_history = st.container()


with chat_history:
    for message in st.session_state.messages:
        if message['role'] == 'user':
            st.markdown(f'<div class="chat-container chat-user">User: {message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-container chat-bot">Chatbot: {message["content"]}</div>', unsafe_allow_html=True)
