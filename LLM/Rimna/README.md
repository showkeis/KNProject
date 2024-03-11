---
license: mit
datasets:
- conceptual_12m
- HuggingFaceM4/COCO
- visual_genome
language:
- ja
- en
---


# bilingual-gpt-neox-4b-minigpt4

![rinna-icon](./rinna.png)

# Overview
This repository provides an English-Japanese bilingual multimodal conversational model like MiniGPT-4 by combining GPT-NeoX model of 3.8 billion parameters and BLIP-2.

The model is based on [`rinna/bilingual-gpt-neox-4b`](https://huggingface.co/rinna/bilingual-gpt-neox-4b) and [BLIP-2](https://huggingface.co/docs/transformers/main/model_doc/blip-2).

* **Model architecture**

    Similar with [BLIP-2](https://huggingface.co/docs/transformers/main/model_doc/blip-2) and [Vision-CAIR/MiniGPT-4](https://huggingface.co/Vision-CAIR/MiniGPT-4), the model consists of an LLM, vision-encoder with ViT and Q-Former, and linear-layer for connecting the LLM and vision-encoder.

    [`rinna/bilingual-gpt-neox-4b`](https://huggingface.co/rinna/bilingual-gpt-neox-4b) (A 36-layer, 2816-hidden-size transformer-based language model) is used as the LLM instead of [Vicuna](https://github.com/lm-sys/FastChat), which is used in the original [Vision-CAIR/MiniGPT-4](https://huggingface.co/Vision-CAIR/MiniGPT-4).

* **Finetuning**
    
    The finetuning data is the subset of the following datasets.
    * English datasets
      * [Conceptual 12M (CC12M)](https://huggingface.co/datasets/conceptual_12m)
      * [COCO 2014](https://huggingface.co/datasets/HuggingFaceM4/COCO)
      * [Visual Genome](https://huggingface.co/datasets/visual_genome)
    * Japanese datasets
      * [STAIR-captions](https://github.com/STAIR-Lab-CIT/STAIR-captions)
      * [Japanese Visual Genome VQA dataset](https://github.com/yahoojapan/ja-vg-vqa)

    Based on the implementation of [Vision-CAIR/MiniGPT-4](https://huggingface.co/Vision-CAIR/MiniGPT-4), only "first pretraining stage" described in [MiniGPT-4 paper](https://arxiv.org/abs/2304.10592) with the above datasets was conducted, and "second-stage finetuning" proposed in the paper with an aligned image-text dataset created with ChatGPT was NOT conducted.
  
* **Model Series**

    | Variant | Link |
    | :-- | :--|
    | Bilingual 4B MiniGPT4 | https://huggingface.co/rinna/bilingual-gpt-neox-4b-minigpt4 |
    | Bilingual 4B PPO | https://huggingface.co/rinna/bilingual-gpt-neox-4b-instruction-ppo |
    | Bilingual 4B SFT | https://huggingface.co/rinna/bilingual-gpt-neox-4b-instruction-sft |
    | Bilingual 4B 8K | https://huggingface.co/rinna/bilingual-gpt-neox-4b-8k |
    | Bilingual 4B | https://huggingface.co/rinna/bilingual-gpt-neox-4b |
    | Japanese 3.6B PPO | https://huggingface.co/rinna/japanese-gpt-neox-3.6b-instruction-ppo |
    | Japanese 3.6B SFT-v2 | https://huggingface.co/rinna/japanese-gpt-neox-3.6b-instruction-sft-v2 |
    | Japanese 3.6B SFT | https://huggingface.co/rinna/japanese-gpt-neox-3.6b-instruction-sft |
    | Japanese 3.6B | https://huggingface.co/rinna/japanese-gpt-neox-3.6b |

* **Authors**
    
    [Koh Mitsuda](https://huggingface.co/mitsu-koh), [Tianyu Zhao](https://huggingface.co/tianyuz), and [Kei Sawada](https://huggingface.co/keisawada)

---

# I/O Format
A special format has been adopted to construct inputs.
* An input prompt is formatted as a conversation between `ユーザー` and `システム`.
* Each input utterance consists of (1) its speaker (`"ユーザー"` or `"システム"`), (2) a colon (`":"`), (3) a whitespace (`" "`), and (4) utterance text (e.g. `"猫はどんな体勢をしていますか？"`).
* An utterance including an image is formatted as (1) its speaker (`"ユーザー"`), (2) a colon (`":"`), (3) a whitespace (`" "`), (4) a placeholder of the image (`"<Img><ImageHere></Img>"`), (5) another whitespace (`" "`), (6) utterance text (e.g. `"What can you see?"`).  
  * The placeholder (`<ImageHere>`) is automatically replaced with the embedding of an input image in the function `get_context_emb`.
* The input prompt should be ended with `"システム: "` to acknowledge the model to generate a response.
* All the utterances in the input prompt should be separated by a newline `\n`.

Following is an example to construct input from a conversation.
~~~python
prompt = [
    {
        "speaker": "ユーザー",
        "text": "<Img><ImageHere></Img> What can you see?"
    },
    {
        "speaker": "システム",
        "text": "a cat on a table with a laptop"
    },
    {
        "speaker": "ユーザー",
        "text": "猫はどんな体勢をしていますか？"
    },
]
prompt = [
    f"{uttr['speaker']}: {uttr['text']}"
    for uttr in prompt
]
prompt = "\n".join(prompt)
prompt = (
    prompt
    + "\n"
    + "システム: "
)
print(prompt)
"""
ユーザー: <Img><ImageHere></Img> What can you see?
システム: a cat on a table with a laptop
ユーザー: 猫はどんな体勢をしていますか？
システム: 
"""
~~~

---

# How to use the model

**1. Download dependencies**

* BLIP-2 implementation included in MiniGPT-4 is used for inference.
* `customized_mini_gpt4.py` is a script to replace LLM from LLaMA architecture to GPT-NeoX one.
* `checkpoint.pth` is a finetuned weight of the linear layer (file size: 177 MB).

```bash
git clone https://github.com/Vision-CAIR/MiniGPT-4.git
cd ./MiniGPT-4
git checkout 22d8888 # latest version as of July 31, 2023.
wget https://huggingface.co/rinna/bilingual-gpt-neox-4b-minigpt4/resolve/main/customized_mini_gpt4.py
wget https://huggingface.co/rinna/bilingual-gpt-neox-4b-minigpt4/resolve/main/checkpoint.pth
```

**2. Inference**

Please run this script in `MiniGPT-4` directory.

~~~~python
import torch
import requests
from PIL import Image
from minigpt4.processors.blip_processors import Blip2ImageEvalProcessor
from customized_mini_gpt4 import CustomizedMiniGPT4

ckpt_path = "./checkpoint.pth"

model = CustomizedMiniGPT4(gpt_neox_model="rinna/bilingual-gpt-neox-4b")
tokenizer = model.gpt_neox_tokenizer

if torch.cuda.is_available():
    model = model.to("cuda")

if ckpt_path is not None:
    print("Load BLIP2-LLM Checkpoint: {}".format(ckpt_path))
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt['model'], strict=False)

vis_processor = Blip2ImageEvalProcessor()

image_url = "https://huggingface.co/rinna/bilingual-gpt-neox-4b-minigpt4/resolve/main/sample.jpg"
raw_image = Image.open(requests.get(image_url, stream=True).raw).convert('RGB')
image = vis_processor(raw_image).unsqueeze(0).to(model.device)
image_emb = model.encode_img(image)

embs = model.get_context_emb(prompt, [image_emb])

output_ids = model.gpt_neox_model.generate(
    inputs_embeds=embs,
    max_new_tokens=512,
    do_sample=True,
    temperature=1.0,
    top_p=0.85,
    pad_token_id=tokenizer.pad_token_id,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id
)

output = tokenizer.decode(output_ids.tolist()[0], skip_special_tokens=True)
print(output)
"""横になっています。"""
~~~~

---

# Acknowledgement

* [Vision-CAIR/MiniGPT-4](https://huggingface.co/Vision-CAIR/MiniGPT-4)
  * [BLIP-2](https://huggingface.co/docs/transformers/main/model_doc/blip-2)
  * [Lavis](https://github.com/salesforce/LAVIS)

# Licenese
[The MIT license](https://opensource.org/licenses/MIT)