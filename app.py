import gradio as gr
from llama_cpp import Llama
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

# Initialize model
model = Llama(
    model_path="/home/azureuser/models/phi-35-mini.gguf",
    n_ctx=26000,  # ideally 128000, 26k takes around 10GB
    n_threads=4  # ideally 4
)

# Simple chat interface
def chat(message, history):
    response = model(
        message,
        max_tokens=512,
        temperature=0.7
    )
    return response['choices'][0]['text']

# Create feedback function
def save_feedback(feedback, conversation_id):
    with open('feedback.txt', 'a') as f:
        f.write(f"Conversation ID: {conversation_id}, Feedback: {feedback}\n")
    return "Thank you for your feedback!"

# Create Gradio interface
with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.Button("Clear")
    
    # Feedback components
    with gr.Row():
        feedback = gr.Textbox(label="Feedback")
        submit_feedback = gr.Button("Submit Feedback")
    
    msg.submit(chat, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)
    submit_feedback.click(save_feedback, [feedback, gr.State("")], gr.Textbox())

demo.launch(server_name='0.0.0.0', server_port=7860)
