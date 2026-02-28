from llama_cpp import Llama
import json
import re
import time

# --- 配置 ---
model_path = "./finetune_model/qwen3_0.6B_q4_k_m.gguf" 
UNIFIED_INSTRUCTION = "智能家居中控：提取用户指令中的实体与意图，输出标准的JSON控制代码。"

# --- 加载模型 ---
print("正在加载 GGUF 模型...")
llm = Llama(
    model_path=model_path,
    n_ctx=512,        # 稍微调小一点，够用就行
    n_gpu_layers=0, 
    n_threads=4,       # 树莓派5 物理核心数
    n_batch=512,       
    use_mmap=False,    # 【修改点】关闭 mmap，强制加载进内存，避免 SD 卡慢速影响//预热效果
    verbose=False      
)

def predict(user_input, is_warmup=False):
    messages = [
        {"role": "system", "content": "智能家居中控：提取用户指令中的实体与意图，输出标准的JSON控制代码。"},
        {"role": "user", "content": f"任务：{UNIFIED_INSTRUCTION}\n指令：{user_input}"}
    ]
    
    start_time = time.time()
    
    output = llm.create_chat_completion(
        messages=messages,
        max_tokens=512, 
        temperature=0.1,
    )
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # 获取详细耗时
    timing = output['usage']
    prompt_tokens = timing['prompt_tokens']
    completion_tokens = timing['completion_tokens']
    
    # 注意：llama-cpp-python 的 output 对象里其实不直接包含 eval_time，
    # 我们主要靠总时间来判断，或者开启 verbose=True 看底层日志。
    # 这里我们主要看第二次运行的总时间。

    prefix = "[预热]" if is_warmup else "[正式]"
    print(f"{prefix} 耗时: {total_time:.4f} 秒 | 生成: {completion_tokens} tokens")
    
    return output['choices'][0]['message']['content']

# --- 1. 预热 (Warm-up) ---
# 这一步非常重要，让内存和 CPU 准备好
print("\n🔥 正在预热 (Warm-up)... 第一次运行通常较慢")
for i in range(10):
    print(f"{i+1}/10 预热中...", end='\r')
    predict("我想要打开客厅的灯", is_warmup=True)

# --- 2. 正式测试 ---
print("\n=== 🚀 正式测试 (真实速度) ===")
user_text = "把客厅灯关了，顺便打开空调"
print(f"指令: {user_text}")#
# print(f"指令：{user_text}")

# 运行第一次正式测试
result = predict(user_text)
print(f"输出: {result}")

# 运行第二次正式测试 (验证稳定性)

print("\n--- 再次测试 ---")
result_2 = predict("卧室太热了，调到24度")
print(f"指令: 卧室太热了，调到24度")
print(f"输出: {result_2}")
print("\n=== 测试结束 ===\n")

