import flask
from flask import Flask
import typing
from threading import Thread
import os

import torch
from transformers import LlamaForCausalLM, LlamaTokenizer, PreTrainedTokenizerBase, TextIteratorStreamer

MODEL_NAME = 'meta-llama/Llama-2-7b-chat-hf'
HUGGINGFACE_SECRET_NAME = os.environ.get("HF_TOKEN")

app = Flask(__name__)

VOLUME_NAME = 'volume'

tokenizer: typing.Optional[PreTrainedTokenizerBase] = None
model = None

def prepare_model():
    volume = VOLUME_NAME
    huggingface_token = HUGGINGFACE_SECRET_NAME

    model_dir = volume

    global model
    global tokenizer
    
    #model = LlamaForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float16, local_files_only=True)
    model = LlamaForCausalLM.from_pretrained(MODEL_NAME,
                                             token=huggingface_token,
                                             cache_dir=model_dir,
                                             torch_dtype=torch.float16,
                                             local_files_only=True)
    
    #tokenizer = LlamaTokenizer.from_pretrained(model_dir, local_files_only=True)
    tokenizer = LlamaTokenizer.from_pretrained(MODEL_NAME,
                                               token=huggingface_token,
                                               cache_dir=model_dir,
                                               local_files_only=True)
    
    if torch.cuda.is_available():
        model.cuda(0)

    


# service entrypoint
# api service url looks https://everai.expvent.com/api/routes/v1/default/llama2-7b-chat/chat
# for test local url is http://127.0.0.1:8866/chat
@app.route('/chat', methods=['GET','POST'])
def chat():
    if flask.request.method == 'POST':
        data = flask.request.json
        prompt = data['prompt']
    else:
        prompt = flask.request.args["prompt"]

    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    input_ids = input_ids.to('cuda:0')
    output = model.generate(input_ids, max_length=256, num_beams=4, no_repeat_ngram_size=2)
    response = tokenizer.decode(output[0], skip_special_tokens=True)

    text = f'{response}'
    
    # return text with some information
    resp = flask.Response(text, mimetype='text/plain', headers={'x-prompt-hash': 'xxxx'})
    return resp

@app.route('/sse', methods=['GET','POST'])
def sse():
    if flask.request.method == 'POST':
        data = flask.request.json
        prompt = data['prompt']
    else:
        prompt = flask.request.args["prompt"]

    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    input_ids = input_ids.to('cuda:0')

    streamer = TextIteratorStreamer(
        tokenizer, timeout=600.0, skip_prompt=True, skip_special_tokens=True
    )

    generate_args = [input_ids]

    generate_kwargs = dict(
        streamer=streamer,
        max_new_tokens=250,
        do_sample=True,
        top_p=0.95,
        temperature=float(0.8),
        top_k=1,
    )

    t = Thread(target=model.generate, args=generate_args, kwargs=generate_kwargs)
    def generator():
        for text in streamer:
            yield text
    t.start()

    # return active messages from the server to the client
    resp = flask.Response(generator(), mimetype='text/event-stream', headers={})

    return resp

@app.route('/healthy-check', methods=['GET'])
def healthy():
    resp = 'service is ready'
    return resp

if __name__ == '__main__':
    prepare_model()
    app.run(host="0.0.0.0", debug=False, port=8866)
