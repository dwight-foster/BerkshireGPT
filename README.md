# BerkshireGPT: A Large Language Model Trained to Value Invest 

---

## Introduction
This is a 7 billion parameter language model based on the Llama2 architecture. It was trained on the 
Berkshire Hathaway annual letters to shareholders from 2004 to 2021. The model was trained using Lora fine-tuning.

[BerkshireGPT Model Card](https://huggingface.co/dwightf/berkshireGPT)

## Training Data and Process
The model was trained on the annual letters to shareholders from 2004 to 2021. The letters were obtained from the Berkshire Hathaway website. The letters were preprocessed to remove any non-ASCII characters and then tokenized using the Hugging Face tokenizers library. The model was trained using the Lora fine-tuning method. The model was trained on Kaggle using their notebooks and available GPUs. It took about a hour and a half to train for 300 steps. 

## Usage
The model can be used for a variety of tasks such as text generation, summarization, and question answering. The model can be used with the Hugging Face Transformers library or mlx if you have an apple computer.

The inference.ipynb notebook can run on any device but is optimized for cuda gpus. 


```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextStreamer

name = "dwightf/BerkshireGPT"
     
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(name, quantization_config=bnb_config)
tokenizer = AutoTokenizer.from_pretrained(name)
```

Scripts are available in the `scripts` folder to help with using the model. There are inference and RAG scripts.

You can run inference with `BitsAndBytes` quantization on a GPU with only 8GB of memory

You can run the gradio demo with the following command:

```bash
python gradio_scripts/run.py
```

## Future Work

Benchmark the model

Create rag and gradio optimized for apple mlx.

I will also be releasing colab notebooks soon and hopefully the dataset I used.


## Acknowledgements
I would like to thank the creators of [this](https://github.com/brevdev/notebooks/tree/main) repository for helping me fine-tune the model.

I would also like to thank the creators of [this](https://github.com/nicknochnack/Llama2RAG/blob/main/llama2%20notebook.ipynb) repository for the RAG code.

I would also like to thank the creators of the llm I used as a base model found [here](https://huggingface.co/FinGPT/fingpt-forecaster_dow30_llama2-7b_lora). 
## Contact
If you have any questions or would like to collaborate, please feel free to reach out to me at dwightf404@gmail.com

## License
This model is licensed under the MIT License. You may use it for commercial and non-commercial use.

## Reminder
This model is just for testing and educational purposes. It should not be used for making financial decisions and is not financial advice.
