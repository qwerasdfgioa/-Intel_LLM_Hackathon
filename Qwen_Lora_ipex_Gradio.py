import os
import torch
import time
from transformers import AutoTokenizer
from ipex_llm.transformers import AutoModelForCausalLM
from peft import PeftModel
import gradio as gr
from threading import Event

# 设置OpenMP线程数为8, 优化CPU并行计算性能
os.environ["OMP_NUM_THREADS"] = "8"

# 加载基础模型和分词器
base_model_name = "Qwen2-1-5B-Instruct"  # 替换为你的基础模型名称
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)

# 加载LoRA微调后的权重
lora_checkpoint = "./checkpoint-781"
lora_model = PeftModel.from_pretrained(model, lora_checkpoint)

# 创建一个停止事件，用于控制生成过程的中断
stop_event = Event()

# 定义用户输入处理函数
def user(user_message, history):
    return "", history + [[user_message, None]]

# 定义机器人回复生成函数
def bot(history):
    stop_event.clear()
    prompt = history[-1][0]
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to('cpu')

    print(f"\n用户输入: {prompt}")
    print("模型输出: ", end="", flush=True)
    start_time = time.time()

    with torch.inference_mode():
        generated_ids = lora_model.generate(model_inputs.input_ids, max_new_tokens=512)

        processed_generated_ids = []
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids):
            input_length = len(input_ids)
            new_tokens = output_ids[input_length:]
            processed_generated_ids.append(new_tokens)
        generated_ids = processed_generated_ids

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    history[-1][1] = response
    end_time = time.time()
    print(f"\n生成完成，用时: {end_time - start_time:.2f} 秒")

    return history

def stop_generation():
    stop_event.set()

with gr.Blocks() as demo:
    gr.Markdown("# 医疗助手")
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.Button("清除")
    stop = gr.Button("停止生成")

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, chatbot, chatbot
    )
    clear.click(lambda: None, None, chatbot, queue=False)
    stop.click(stop_generation, queue=False)

if __name__ == "__main__":
    print("启动 Gradio 界面...")
    demo.queue()
    demo.launch(root_path='/dsw-607239/proxy/7860/')
