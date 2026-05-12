from openai import OpenAI

client = OpenAI()

# 1. 上传文件
file = client.files.create(
    file=open("/Users/franky/Downloads/prune-to-prosper-main/paper/main.pdf", "rb"),
    purpose="user_data"
)

# 2. 在 Responses API 中引用 file_id
response = client.responses.create(
    model="gpt-5.5",
    input=[
        {
            "role": "user",
            "content": [
                {
                    "type": "input_file",
                    "file_id": file.id
                },
                {
                    "type": "input_text",
                    "text": "严格按照EMNLP的要求审稿这篇论文。"
                }
            ]
        }
    ],
    reasoning={
        "effort": "xhigh"
    }
)

print(response.output_text)