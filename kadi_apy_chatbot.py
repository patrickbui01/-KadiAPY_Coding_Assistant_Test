import os
import json
import gradio as gr
from huggingface_hub import HfApi, login
from dotenv import load_dotenv

from llm import get_groq_llm
from vectorstore import get_chroma_vectorstore
from embeddings import get_SFR_Code_embedding_model
from kadiApy_ragchain import KadiApyRagchain

load_dotenv()

vectorstore_path = "data/vectorstore"

GROQ_API_KEY = os.environ["GROQ_API_KEY"]
HF_TOKEN = os.environ["HF_Token"]

login(HF_TOKEN)
hf_api = HfApi()



class KadiBot:
    def __init__(self, llm, vectorstore):
        self.kadiAPY_ragchain = KadiApyRagchain(llm, vectorstore)

    def handle_chat(self, chat_history):
        if not chat_history:
            return chat_history
        
        user_query = chat_history[-1][0]
        response = self.kadiAPY_ragchain.process_query(user_query, chat_history)
        chat_history[-1] = (user_query, response)

        return chat_history




def add_text_to_chat_history(chat_history, user_input):
    chat_history = chat_history + [(user_input, None)]
    return chat_history, ""


def show_history(chat_history):
    return chat_history


def main():
    vectorstore = get_chroma_vectorstore(get_SFR_Code_embedding_model(), vectorstore_path)
    llm = get_groq_llm("qwen-2.5-coder-32b", "0.0", GROQ_API_KEY)
    
    kadi_bot = KadiBot(llm, vectorstore)
    
    with gr.Blocks() as demo:
        gr.Markdown("## KadiAPY - AI Coding-Assistant")
        gr.Markdown("AI Coding-Assistnat for KadiAPY based on RAG architecture powered by LLM")
        
        # Create a state for session management
        chat_history = gr.State([])

        with gr.Tab("KadiAPY - AI Assistant"):
            with gr.Row():
                with gr.Column(scale=10):
                    chatbot = gr.Chatbot([], elem_id="chatbot", label="Kadi Bot", bubble_full_width=False, show_copy_button=True, height=600)
                    user_txt = gr.Textbox(label="Question", placeholder="Type in your question and press Enter or click Submit")
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            submit_btn = gr.Button("Submit", variant="primary")
                        with gr.Column(scale=1):
                            clear_input_btn = gr.Button("Clear Input", variant="stop")
                        with gr.Column(scale=1):
                            clear_chat_btn = gr.Button("Reset Chat", variant="stop")  # New button to clear chat history
                    
                    gr.Examples(
                        examples=[
                            "Write me a python script with which can convert plain JSON to a Kadi4Mat-compatible extra metadata structure",
                            "I need a method to upload a file to a record. The id of the record is 3",
                        ],
                        inputs=user_txt,
                        outputs=chatbot,
                        fn=add_text_to_chat_history,
                        label="Try asking...",
                        cache_examples=False,
                        examples_per_page=3,
                    )
        
        # Use the state to persist chat history between interactions
        user_txt.submit(add_text_to_chat_history, [chat_history, user_txt], [chat_history, user_txt]).then(show_history, [chat_history], [chatbot])\
                  .then(kadi_bot.handle_chat, [chat_history], [chatbot])          
        submit_btn.click(add_text_to_chat_history, [chat_history, user_txt], [chat_history, user_txt]).then(show_history, [chat_history], [chatbot])\
                  .then(kadi_bot.handle_chat, [chat_history], [chatbot])        
        clear_input_btn.click(lambda: ("",), [], [user_txt])
        clear_chat_btn.click(lambda: ([], ""), [], [chat_history, chatbot]) 


    demo.launch()


if __name__ == "__main__":
    main()
