from vllm import LLM, SamplingParams

llm = LLM(
    model="/dev/shm/Qwen3-30B-A3B", # Hoặc đường dẫn local của bạn
    gpu_memory_utilization=0.9,               # Dùng 85% VRAM
    cpu_offload_gb=10,                          # Giữ ở 0 nếu model vừa VRAM. Đặt > 0 nếu muốn offload weights.
    tensor_parallel_size=1, 
    enforce_eager= True,
)

# 2. Khởi tạo tham số sinh văn bản
sampling_params = SamplingParams(
    temperature=0,  #greedy
    # top_p=0.95, 
    max_tokens=512,
)

# 3. Chuẩn bị danh sách prompts (có thể là hàng nghìn prompts, vLLM sẽ tự động continuous batching)
prompts = [
    "[INST] Giải thích cơ chế Self-Attention trong Transformer. [/INST]",
    "[INST] Viết một đoạn code Python dùng hybrid search kết hợp BM25 và Vector. [/INST]",
    "[INST] Sự khác biệt giữa Speculative Decoding và tiêu chuẩn là gì? [/INST]"
]

# 4. Truyền prompt vào model
# Hàm generate() sẽ chạy toàn bộ batch và trả về list kết quả
outputs = llm.generate(prompts, sampling_params)

# 5. Xử lý kết quả đầu ra
for output in outputs:
    prompt = output.prompt
    # outputs[0] chứa câu trả lời tốt nhất (nếu n=1 trong SamplingParams)
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}")
    print(f"Generated text: {generated_text!r}")
    print("-" * 50)