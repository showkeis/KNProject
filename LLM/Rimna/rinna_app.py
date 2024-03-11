import streamlit as st
import torch
from PIL import Image
from customized_mini_gpt4 import CustomizedMiniGPT4
from minigpt4.processors.blip_processors import Blip2ImageEvalProcessor

# Streamlitアプリのタイトル
st.title("MiniGPTモデルによる会話")

# モデルとトークナイザの初期化
model = CustomizedMiniGPT4(gpt_neox_model="rinna/bilingual-gpt-neox-4b")
tokenizer = model.gpt_neox_tokenizer
vis_processor = Blip2ImageEvalProcessor()

# CUDAの利用可能性を確認
if torch.cuda.is_available():
    model = model.to("cuda")

# チェックポイントの読み込み
ckpt_path = "./checkpoint.pth"
if ckpt_path is not None:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt['model'], strict=False)

# Streamlit UI
user_input = st.text_input("テキストを入力してください", "")
uploaded_file = st.file_uploader("画像をアップロードしてください（オプション）", type=["png", "jpg", "jpeg"])

if st.button('返信を生成'):
    prompt = f"ユーザー: {user_input}\nシステム: "
    embs = None  # コンテキストエンベッディング用の変数を初期化

    if uploaded_file is not None:
        # 画像の処理
        image = Image.open(uploaded_file).convert('RGB')
        image = vis_processor(image).unsqueeze(0).to(model.device)
        image_emb = model.encode_img(image)
        # コンテキストエンベッディングの取得（画像あり）
        embs = model.get_context_emb(prompt, [image_emb])
    else:
        # コンテキストエンベッディングの取得（テキストのみ）
        embs = model.get_context_emb(prompt, [])

    # モデルによる応答の生成
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
    st.write(output)
