import streamlit as st
import time
import torch
from classes import TorchTransformer
import torchtext
import pandas as pd
import re
import torch.utils.data as data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# load vocab and tokenizer
transformer_state_dict = torch.load('./model/torchModel.pth')
vocab_ques = torch.load('./model/questions_vocab.pth')
vocab_ans = torch.load('./model/answers_vocab.pth')
tokenizer = torch.load('./model/tokenizer.pth')

# model hyperparameters
ques_vocab_size = len(vocab_ques)
ans_vocab_size = len(vocab_ans)
d_model = 512
n_heads = 8
n_layers = 6
d_ff = 2048
max_seq_length = 128
dropout = 0.1


# load model
transformer = transformer = TorchTransformer(ques_vocab_size, ans_vocab_size, d_model, n_heads, n_layers, d_ff, max_seq_length, dropout)
transformer.load_state_dict(transformer_state_dict)
transformer = transformer.to(device)

# cleaner
def cleaner(text):
  text = text.lower()
  text = re.sub(r'\W', ' ', text)
  text = text.strip()
  return text

# question dataframe
# question = 'What causes Glaucoma ?'
 
def answer(model, tokenizer, vocab, question, max_length=128):
    model.eval() 
    question = cleaner(question)
    question_tokens = tokenizer(question)
    question_encoded = [vocab["<sos>"]] + vocab(question_tokens) + [vocab["<eos>"]]
    question_tensor = torch.tensor(question_encoded, dtype=torch.long).unsqueeze(0).to(device)
    answer_encoded = [vocab["<sos>"]]
    answer_tensor = torch.tensor(answer_encoded, dtype=torch.long).unsqueeze(0).to(device)
    for _ in range(max_length):
        with torch.no_grad():
            output = model(question_tensor, answer_tensor)
            output_list = output.tolist()
            next_token_id = output.argmax(dim=-1)[:, -1].item()
        if next_token_id == vocab["<eos>"]:
            break
        answer_encoded.append(next_token_id)
        answer_tensor = torch.tensor(answer_encoded, dtype=torch.long).unsqueeze(0).to(device)
    answer_tokens = vocab.lookup_tokens(answer_encoded[1:])
    answer = " ".join(answer_tokens)
    return answer

# answer(transformer, tokenizer, vocab_ans, question)
    
        
st.title('Doctor Chatbot')

if "messages" not in st.session_state:
    st.session_state.messages = []
    
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])
        

def chat_answer_view(prompt):    
    pred_text = answer(transformer, tokenizer, vocab_ans, prompt)
    for word in pred_text.split():
        yield word + " "
        time.sleep(0.05)
        
# user input
if prompt := st.chat_input('Say something ... '):
    with st.chat_message('user'):
        st.markdown(prompt)
    st.session_state.messages.append({
        'role': 'user',
        'content': prompt
        })
    
    with st.chat_message('assistant'):
        response = st.write_stream(chat_answer_view(prompt))
    st.session_state.messages.append({
        'role': 'assistant',
        'content': response
        })
    

