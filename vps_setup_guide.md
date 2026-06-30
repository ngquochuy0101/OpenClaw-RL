# 🛠️ Cài Đặt Môi Trường OpenClaw-RL trên Ubuntu VPS (2x RTX 3060)

## Bước 1: Clone Source Code

```bash
mkdir -p ~/Projects
cd ~/Projects
git clone https://github.com/ngquochuy0101/OpenClaw-RL.git
cd OpenClaw-RL
```

## Bước 2: Cài đặt thư viện & Fix lỗi

### 1. Sửa lỗi build C++ (Lỗi thiếu Header)
Giải quyết lỗi: `"fatal error: cudnn.h / nccl.h: No such file or directory"`
```bash
conda install -c conda-forge cudnn=9.10.* nccl -y
APEX_CPP_EXT=1 APEX_CUDA_EXT=1 python -m pip install -v --no-build-isolation .
```

### 2. Cài đặt Flash Attention & Sửa lỗi OOM
Giải quyết lỗi: `"Killed" (OOM - hết RAM)` khi build Flash Attention. Ta cần giới hạn MAX_JOBS.
```bash
export MAX_JOBS=2
python -m pip install --no-build-isolation -v flash-attn==2.7.4.post1
```

### 3. Cài đặt FlashInfer, Megatron-Bridge & Transformer Engine
```bash
python -m pip install "flashinfer-jit-cache==0.6.3" --index-url https://flashinfer.ai/whl/cu129
python -m pip install "megatron-bridge @ git+https://github.com/fzyzcjy/Megatron-Bridge.git@35b4ebfc486fb15dcc0273ceea804c3606be948a" --no-build-isolation
export NVTE_FRAMEWORK=pytorch
python -m pip install --no-build-isolation "transformer_engine[pytorch,core_cu12]==2.10.0"
```

---

## Bước 3: Tải Model 3B/4B

Sử dụng lệnh `huggingface-cli` để tải trực tiếp model về VPS:
```bash
python -m pip install -U "huggingface_hub[cli]"
mkdir -p ~/Projects/OpenClaw-RL/models
huggingface-cli download Qwen/Qwen2.5-3B-Instruct --local-dir ~/Projects/OpenClaw-RL/models/Qwen2.5-3B-Instruct
```

---

## Bước 4: Cấu hình Script chạy cho 2 GPU (Tránh OOM) ⚠️

Bạn cần tạo một file script riêng biệt được tối ưu cho cấu hình máy VPS của bạn (2x RTX 3060, tổng 24GB VRAM).

Chạy lệnh sau để tạo file cấu hình:
```bash
cat << 'EOF' > ~/Projects/OpenClaw-RL/openclaw-combine/run_qwen_3b_vps.sh
#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1
export HF_CKPT="$HOME/Projects/OpenClaw-RL/models/Qwen2.5-3B-Instruct"
export SAVE_CKPT="$HOME/Projects/OpenClaw-RL/ckpt/qwen2.5-3b-lora-vps"

# Cấu hình cho 2 GPUs
export NUM_GPUS=2
export ACTOR_GPUS=1
export ROLLOUT_GPUS=1
export PRM_GPUS=0

# Tensor Parallelism
export TP=1

# Tối ưu bộ nhớ
export MAX_TOKENS_PER_GPU=4096
export MICRO_BATCH_SIZE=1
export GLOBAL_BATCH_SIZE=1

echo "🚀 Bắt đầu khởi chạy RL Server với cấu hình 2 GPU..."
bash ../openclaw-combine/run_qwen3_4b_openclaw_combine_lora.sh
EOF

chmod +x ~/Projects/OpenClaw-RL/openclaw-combine/run_qwen_3b_vps.sh
```

---

## Bước 5: Khởi động RL Server & Fix lỗi Ray/Gray

**Cách fix lỗi Ray (nếu gặp lỗi khi chạy server hoặc server bị treo/gray screen):**
Đôi khi framework Ray bị kẹt tiến trình cũ dẫn đến việc khởi động script báo lỗi hoặc bị kẹt. Hãy dọn dẹp tiến trình cũ trước khi chạy:
```bash
ray stop
ray start --head
```

**Khởi động Server:**
```bash
cd ~/Projects/OpenClaw-RL/slime
bash ../openclaw-combine/run_qwen_3b_vps.sh
```
Đợi cho đến khi bạn thấy log báo server SGLang đã khởi động thành công ở cổng `30000`.

---

## Bước 6: Sử dụng Pinggy để lấy API Public (Vì VPS bị chặn IP trực tiếp)

Do server của bạn không cho phép truy cập IP trực tiếp từ bên ngoài, bạn có thể dùng **Pinggy** để tạo đường hầm (tunnel) đưa port `30000` ra public internet.

Mở một **Terminal/SSH session mới** vào VPS và chạy lệnh:
```bash
ssh -p 443 -R0:localhost:30000 a.pinggy.io
```
Ngay sau đó, màn hình terminal sẽ hiển thị một đường link HTTP/HTTPS công khai (Ví dụ: `https://xxxx-xxxx.a.free.pinggy.link`). Hãy copy đường link HTTPS này, đây chính là API URL của bạn.

---

## Bước 7: Cấu hình OpenClaw Client (Máy cá nhân)

Trên máy **Laptop Windows/Mac** của bạn:

1. Bạn **KHÔNG CẦN** chạy bất kỳ script WSL2 hay tải model nào nữa.
2. Mở file cấu hình của OpenClaw (Ví dụ: `openclaw.json` hoặc phần thiết lập provider trong UI).
3. Đặt `baseUrl` bằng đường link Pinggy bạn vừa lấy được (nhớ thêm `/v1` vào cuối).

**Ví dụ sửa file `openclaw.json`:**
```json
{
  "models": {
    "providers": {
      "qwen": {
        "baseUrl": "https://xxxx-xxxx.a.free.pinggy.link/v1",
        "apiKey": "no-auth-needed",
        "api": "openai-completions",
        "models": [
          {
            "id": "qwen2.5-3b-instruct",
            "name": "Qwen2.5 3B VPS"
          }
        ]
      }
    }
  }
}
```

4. Bắt đầu chat và dùng agent trên Laptop! Mọi request sẽ tự động được gửi qua API Pinggy về VPS để xử lý và huấn luyện tự động.
