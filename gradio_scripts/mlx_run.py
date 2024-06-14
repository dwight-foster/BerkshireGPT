import gradio as gr
import random
import time
import torch
from mlx_lm import load
from engine import PDFReader
model, tokenizer = load("dwightf/BerkshireGPTMLX")
pdf_reader = PDFReader(model, tokenizer, mlx=True)
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Row():
                checkbox = gr.CheckboxGroup(label="Stock Data", choices=["Financials", "News", "Analysts"])
                ticker = gr.Textbox(label="Enter Ticker")
            file = gr.File(file_types=[".pdf"], file_count="multiple")
        with gr.Column(scale=2):
            chatbot = gr.Chatbot()

            msg = gr.Textbox()
            clear = gr.Button("Clear")


    def user(user_message, history):
        return "", history + [[user_message, ""]]


    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        pdf_reader.query, [chatbot, file, checkbox, ticker], chatbot
    )
    clear.click(lambda: None, None, chatbot, queue=False)
    file.clear(pdf_reader.clear, queue=False)

demo.queue()
demo.launch()
