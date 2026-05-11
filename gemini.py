import google.generativeai as genai
import os

# 配置 API Key
genai.configure(api_key="AIzaSyDtlGPwYyHZAROUtqxdg_xSssgm_rSva4M")

# 选择模型
model = genai.GenerativeModel("gemini-2.5-pro")

# 发送文本消息
response = model.generate_content("请用一句话介绍你自己")
print(response.text)

# 多模态输入（文本 + 图片）
# img = PIL.Image.open('image.jpg')
# response = model.generate_content(["这张图片里有什么？", img])
# print(response.text)
