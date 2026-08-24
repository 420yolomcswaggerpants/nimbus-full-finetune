import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM

st.title("☕ Nimbus Coffee - Full Fine-Tuned Assistant")
st.caption("Qwen 2.5 0.5B full fine-tuned on 80 Q&A pairs")

@st.cache_resource
def load_model():
    model_name = "420yolomcswaggerpants/nimbus-full-finetune-0.5b"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

user_input = st.chat_input("Ask about Nimbus Coffee...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)
    
    with st.chat_message("assistant"):
        prompt = f"### Instruction:\n{user_input}\n\n### Response:\n"
        inputs = tokenizer(prompt, return_tensors="pt")
        
        with st.spinner("Thinking..."):
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.3,
                do_sample=True
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.split("### Response:\n")[-1].strip()
        st.write(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})