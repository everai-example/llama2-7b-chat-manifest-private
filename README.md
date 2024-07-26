#  Llama-2(7B) Chat by manifest yaml file

In this example, we can implement an AIGC(AI generated content) online question and answer service based on the `Llama-2(7B)` model on the [EverAI](https://everai.expvent.com) platform by defining a manifest yaml file. This method allows you to deploy your application to the EverAI platform without importing EverAI SDK code into your existing business code.  

## Login EVERAI CLI

Create a directory for your app firstly. In your app directory, you should login by token you got in [EverAI](https://everai.expvent.com).     
    
```bash
everai login --token <your token>
```

## Prepare volume

Create a data volume to store model files.  

```bash
everai volume create models--meta-llama--llama-2-7b-chat-hf
```

You can get the local path of the volume `models--meta-llama--llama-2-7b-chat-hf` through the `everai volume get` command. After entering the local path of the volume, you can copy your model files to that path.

Use the `everai volume push` command to push the model files in the local environment of the data volume `models--meta-llama--llama-2-7b-chat-hf` to the cloud.

```bash
everai volume push models--meta-llama--llama-2-7b-chat-hf
```

In the current application directory, create a soft link to mount the local path of the volume `models--meta-llama--llama-2-7b-chat-hf`.  

```bash
ln -s /root/.cache/everai/volumes/UoswTkyjkPZU3qb27s4B9E volume
```

## Write your app code in python

### Load model

There is an example code in [app.py](https://github.com/everai-example/llama2-7b-chat-manifest-private/blob/main/app.py).  

When you use `python run` to debug the sample code to load the model in the local environment, the model file will be read from the private volume `models--meta-llama--llama-2-7b-chat-hf` in the local environment. If your local debugging environment has GPU resources, the system will successfully execute `model.cuda(0)`.

```python
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer, PreTrainedTokenizerBase, TextIteratorStreamer

MODEL_NAME = 'meta-llama/Llama-2-7b-chat-hf'
HUGGINGFACE_SECRET_NAME = os.environ.get("HF_TOKEN")

app = Flask(__name__)

VOLUME_NAME = 'volume'

def prepare_model():
    volume = VOLUME_NAME
    huggingface_token = HUGGINGFACE_SECRET_NAME

    model_dir = volume

    global model
    global tokenizer
    
    model = LlamaForCausalLM.from_pretrained(MODEL_NAME,
                                             token=huggingface_token,
                                             cache_dir=model_dir,
                                             torch_dtype=torch.float16,
                                             local_files_only=True)
    
    tokenizer = LlamaTokenizer.from_pretrained(MODEL_NAME,
                                               token=huggingface_token,
                                               cache_dir=model_dir,
                                               local_files_only=True)
    
    if torch.cuda.is_available():
        model.cuda(0)
```

### Generate inference service
Aftering loading `Llama-2(7B)` model, now you can write your Python code that uses `flask` to implement the inference online service of AIGC(AI generated content).    

```python
import flask
import typing

tokenizer: typing.Optional[PreTrainedTokenizerBase] = None
model = None

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
```

### Generate readinessProbe service

If readinessProbe is set up, there are no any request be route to this worker before probe status is ready(status code is `200`), otherwise (readinessProbe is not set up), everai platform will route client reqeust to this worker when container is ready, even model files have not been loaded into memory of GPU yet. 

HTTP `get` and `post` probe is the only supported methods now.

```python
@app.route('/healthy-check', methods=['GET'])
def healthy():
    resp = 'service is ready'
    return resp

if __name__ == '__main__':
    prepare_model()
    app.run(host="0.0.0.0", debug=False, port=8866)
```

## Build image
This step will build the container image using `Dockerfile`.    

There is an example code in [Dockerfile](https://github.com/everai-example/llama2-7b-chat-manifest-private/blob/main/Dockerfile).   

You can choose the public image registry to store application image, such as [quay.io](https://quay.io/), [Docker Hub](https://hub.docker.com/), [GitHub Container Registry](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry), [Google Container Registry](https://cloud.google.com/artifact-registry), etc. If you have a self-built image registry and the image can be accessed on the Internet, you can also use it.  

The dependence of this step is docker installed on your machine. It is recommended to use `docker buildx` to build Docker image that support multi-platform architecture, and push the packaged image to your specified registry.

## Create secrets
Secrets are a secure way to add credentials and other sensitive information to the containers your functions run in.  

This step is optional, depending on whether the model and Docker image require security certification.  

You can create and edit secrets on [EverAI](https://everai.expvent.com), or programmatically from Python code.  

In this case, we will create one secret for [quay.io](https://quay.io/). 
```bash
everai secret create your-quay-io-secret-name \
  --from-literal username=<your username> \
  --from-literal password=<your password>
```

## Create configmap
>Optional, but you can use configmap for adjust autoscaling policy after deploying the image. 
```shell
everai configmap create llama2-configmap \ 
  --from-literal min_workers=1 \
  --from-literal max_workers=5 \
  --from-literal min_free_workers=2 \
  --from-literal scale_up_step=1 \
  --from-literal max_idle_time=60
```

## Define manifest file
The manifest file defines various information required to create an EverAI application, including application name, image name, key information, data volume information, etc.  

There is an example code in [app.yaml](https://github.com/everai-example/llama2-7b-chat-manifest-private/blob/main/app.yaml).

## Deploy image
The final step is to deploy your app to everai and keep it running.  

```bash
everai app create --from-file app.yaml
```

After running `everai app list`, you can see the result similar to the following. Note that `CREATED_AT` uses UTC time display.  

If your app's status is `DEPLOYED`, and the number of ready worker containers is equal to the expected number of worker containers, which is `1/1`, it means that your app is deployed successfully.  

```bash
NAME                   NAMESPACE    STATUS    WORKERS    CREATED_AT
---------------------  -----------  --------  ---------  ------------------------
llama2-7b-chat         default      DEPLOYED  1/1        2024-06-19T08:07:24+0000
```

When your app is deployed, you can use `curl` to execute the following request to test your deployed code, and you can see that `Llama-2(7B)` model gives the answers to the question on the console. The following data information is displayed.  

```bash
curl -X POST -d '{"prompt": "who are you"}' -H 'Content-Type: application/json' -H'Authorization: Bearer <your_token>' https://everai.expvent.com/api/routes/v1/<your namespace>/<your app name>/chat
who are you?

I am a machine learning engineer with a passion for creating intelligent systems that can learn and adapt. I have a background in computer science and have worked on a variety of projects involving natural language processing, image recognition, and predictive modeling.
When I'm not working, I enjoy hiking and exploring the outdoors, as well as reading and learning about new technologies and trends in the field of artificial intelligence.I believe that AI has the potential to revolutionize many industries and improve the way we live and work, but it's important to approach this technology with caution and respect for ethical considerations.
```
