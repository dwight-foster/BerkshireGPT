import yfinance as yf
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from transformers import TextIteratorStreamer
from threading import Thread
from llama_index.core import PromptTemplate
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
class PDFReader:
    def __init__(self, model, tokenizer):
        self.reader = None
        self.index = None
        self.rag = False
        self.extra_text = ""
        self.max_history = 5
        self.model = model
        self.tokenizer = tokenizer
        system_prompt = """[INST]<<SYS>>\n You are a value investor giving your advice on stocks. And choosing 
        whether to buy, sell, or hold them. \n\n <</SYS>>"""
        def complete_to_prompt(complete: str) -> str:
            return f"\nQ - {complete}\n[/INST]\nA -"
        llm = HuggingFaceLLM(context_window=4096,
                             max_new_tokens=1024,
                             system_prompt=system_prompt,
                             completion_to_prompt=complete_to_prompt,
                             model=model,
                             tokenizer=tokenizer)
        Settings.llm = llm
        Settings.chunk_size = 1024
        embeddings = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")

        Settings.embed_model = embeddings

    def read_pdf(self, pdf_file):
        names = []
        for file in pdf_file:
            names.append(file.name)
        self.reader = SimpleDirectoryReader(input_files=names).load_data()
        self.index = VectorStoreIndex.from_documents(self.reader)
        self.rag = True

    def format_query(self, query):
        if self.rag:
            return "".join(["".join([query[i][0], query[i][1]])
                            for i in range(max(0, len(query) - self.max_history), len(query))]) + self.extra_text

        else:

            messages = ""
            for i in range(max(0, len(query) - self.max_history), len(query)):

                item = query[i]
                if i == len(query) - 1:
                    messages += "".join([
                                            "[INST]<<SYS>>\n You are a value investor giving your advice on stocks. And choosing whether to buy, sell, or hold them. \n\n <</SYS>>\n",
                                            self.extra_text, "\n Q -", item[0], "\n[/INST]\nA - ", item[1]])
                else:
                    messages += "".join([
                                            "[INST]<<SYS>>\n You are a value investor giving your advice on stocks. And choosing whether to buy, sell, or hold them. \n\n <</SYS>>\n\n Q -",
                                            item[0], "\n[/INST]\nA - ", item[1]])
            return messages

    def query(self, query, file, options, ticker):
        self.checkbox_options(options, ticker)

        if file and not self.rag:
            self.read_pdf(file)

        if self.rag:
            yield from self.rag_query(query)
        else:
            yield from self.chat_query(query)



    def rag_query(self, history):
        messages = self.format_query(history)
        query_engine = self.index.as_query_engine(streaming=True, similarity_top_k=1)
        response = query_engine.query(messages)
        history[-1][1] = ""
        for new_token in response.response_gen:
            if new_token != '\s':
                history[-1][1] += new_token
                yield history

    def chat_query(self, history):
        messages = self.format_query(history)

        model_inputs = self.tokenizer([messages], return_tensors="pt").to("cuda")
        streamer = TextIteratorStreamer(self.tokenizer, timeout=10., skip_prompt=True, skip_special_tokens=True)
        generate_kwargs = dict(
            model_inputs,
            streamer=streamer,
            max_new_tokens=1024,
            do_sample=True,
            top_p=0.95,
            top_k=1000,
            temperature=1.0,
            num_beams=1
        )
        t = Thread(target=self.model.generate, kwargs=generate_kwargs)
        t.start()
        history[-1][1] = ""
        for new_token in streamer:
            if new_token != '<':
                history[-1][1] += new_token
                yield history

    def clear(self):
        self.reader = None
        self.index = None
        self.rag = False

    def checkbox_options(self, options, ticker):
        text = str(ticker) + "\n"
        if len(options) == 0:
            self.extra_text = ""
        if ticker:
            for option in options:
                if option == "Financials":
                    text += "Financials \n"
                    text += self.get_financials(ticker)
                elif option == "News":
                    text += "News \n"
                    text += str(self.get_news(ticker))
                elif option == "Analysts":
                    text += "Analysts \n"
                    text += str(self.get_analysts(ticker))
            self.extra_text = text


    def get_news(self, ticker):
        stock = yf.Ticker(ticker)
        news = stock.news
        return news

    def get_analysts(self, ticker):
        stock = yf.Ticker(ticker)
        analysts = stock.recommendations
        return analysts

    def get_financials(self, ticker):
        financials = ""
        stock = yf.Ticker(ticker)
        financials += "Balance Sheet\n"
        financials += str(stock.balance_sheet)
        financials += "Income Statement\n"
        financials += str(stock.income_stmt)
        financials += "Cash Flow\n"
        financials += str(stock.cashflow)
        return financials