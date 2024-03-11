import requests

def download_file(url, destination_filename):
    response = requests.get(url)
    response.raise_for_status() 
    with open(destination_filename, 'wb') as f:
        f.write(response.content)

# customized_mini_gpt4.py のダウンロード
download_file("https://huggingface.co/rinna/bilingual-gpt-neox-4b-minigpt4/resolve/main/customized_mini_gpt4.py", "customized_mini_gpt4.py")

# checkpoint.pth のダウンロード
download_file("https://huggingface.co/rinna/bilingual-gpt-neox-4b-minigpt4/resolve/main/checkpoint.pth", "checkpoint.pth")
