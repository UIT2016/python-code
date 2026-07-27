import time
from llama_cpp import Llama

model_path = "D:\\code\\LLM\\models\\Mistral-Nemo-Instruct-2407.Q6_K.gguf"
imatrix_path = "D:\\code\\LLM\\models\\Mistral-Nemo-Instruct-2407-GGUF_imatrix.dat"

llm = Llama(
    model_path=model_path,
    n_gpu_layers=20,  # 取消注释使用 GPU 加速
    imatrix=imatrix_path, 
    n_threads=12,
    use_mlock=True,  # 锁定内存以提升性能
    use_mmap=True,   # 使用内存映射文件
    n_ctx=6144            #6GB context
)

input_text = "介绍一下你自己"

start_time = time.time()
output = llm(input_text, max_tokens=1200)
end_time = time.time()

elapsed_time = end_time - start_time

print(f"生成的文本：{output['choices'][0]['text']}")
print(f"耗时：{elapsed_time:.2f} 秒")
