<div align="center">
  <h1 align="center">
    OpenClaw-RL
  </h1>
  <p><b>Dự án OpenClaw-RL - Tối ưu hóa tác tử AI thông qua Reinforcement Learning (RL) và hội thoại tự nhiên</b></p>
</div>

## 💡 Giới thiệu (TL;DR)

**OpenClaw-RL** là một framework học tăng cường (Reinforcement Learning) bất đồng bộ hoàn toàn (fully async), cho phép biến các cuộc trò chuyện hàng ngày thành tín hiệu huấn luyện cho các tác tử AI cá nhân hóa. Nó cũng hỗ trợ việc huấn luyện các tác tử tổng quát với quy mô lớn.

Thay vì dựa trên các dataset được thu thập sẵn theo batch, **OpenClaw-RL** sử dụng cách tiếp cận khác: wrap mô hình tự lưu trữ của bạn (self-hosted) thành một API tương thích với OpenAI, sau đó can thiệp vào các cuộc hội thoại nhiều lượt (multi-turn) và liên tục tối ưu hóa policy trong background — mà không làm gián đoạn quá trình sử dụng của bạn.

## 🚀 Các tính năng chính

- **Kiến trúc 4 thành phần bất đồng bộ hoàn toàn**: OpenClaw-RL tách biệt **phục vụ agent (serving)**, **thu thập rollout**, **đánh giá PRM/judge** và **huấn luyện policy** thành các vòng lặp độc lập.
- **Self-Hosted & Riêng tư**: Toàn bộ hệ thống, từ policy model đến PRM/judge và trainer, đều chạy trên hạ tầng của bạn. Dữ liệu hội thoại hoàn toàn nằm trên máy của bạn mà không cần đến API từ bên thứ ba.
- **Tự động từ Feedback đến Gradient**: Hệ thống tự động:
  - Phân tích hội thoại multi-turn thành các trajectory huấn luyện.
  - Sử dụng các phản hồi từ người dùng hoặc công cụ làm tín hiệu (reward/feedback) "next-state".
  - Chấm điểm PRM/judge ngầm.
  - Gửi dữ liệu sẵn sàng cho bộ huấn luyện.
- **Ba phương pháp tối ưu hóa được tích hợp**:
  - **Binary RL (GRPO)**: Dùng Process Reward Model để chấm điểm và dùng GRPO để tối ưu.
  - **On-Policy Distillation (OPD)**: Trích xuất gợi ý (hint) dựa vào các state tương lai và tạo tín hiệu định hướng chi tiết ở mức độ token.
  - **Hybrid Method (Kết hợp)**: Kết hợp cả hai phương pháp Binary RL và OPD để tận dụng giám sát vô hướng (scalar) và định hướng token, cho ra kết quả mạnh mẽ nhất.

## 🛠 Cài đặt và Sử dụng

### 1. Yêu cầu hệ thống

- **Phần cứng:** Cấu hình multi-GPU (Mặc định 8x GPUs, có thể thay đổi bằng biến môi trường)
- **Phần mềm:** CUDA 12.9, Python 3.12
- **Cấu trúc:** Base RL framework sử dụng `slime`.

### 2. Khởi chạy RL Server

Bạn có thể sử dụng các script có sẵn trong thư mục `openclaw-combine` để chạy mô hình kết hợp (Hybrid):

```bash
cd slime
bash ../openclaw-combine/run_qwen3_4b_openclaw_topk_select.sh
```

Khi server đang chạy, API tương thích với OpenAI sẽ được phục vụ tại:
`http://<HOST_IP>:30000/v1`

### 3. Tích hợp OpenClaw

Để AI có thể tự học qua OpenClaw, bạn cần cấu hình OpenClaw gửi truy vấn tới server RL của bạn:
Cài đặt Extension tại `extensions/rl-training-headers/`.
Sửa tệp `openclaw.json` (hoặc cấu hình model) để trỏ URL đến RL server của bạn:

```json
{
  "models": {
    "providers": {
      "qwen": {
        "baseUrl": "http://<HOST_IP>:30000/v1",
        "apiKey": "apiKey",
        "api": "openai-completions",
        "models": [
          {
            "id": "qwen3-4b",
            "name": "Qwen3 4B RL",
            "contextWindow": 32768,
            "maxTokens": 8192
          }
        ]
      }
    }
  }
}
```

## 🤖 Môi trường Agent Thực tế

OpenClaw-RL hỗ trợ huấn luyện trong nhiều môi trường thực tế khác nhau:

- **Terminal Agent** (`terminal-rl/`): Agent tự tương tác và chạy code trên shell terminal.
- **GUI Agent** (`gui-rl/`): Tương tác trực tiếp trên màn hình, đánh giá qua state đồ họa.
- **SWE Agent** (`swe-rl/`): Agent thực hiện quy trình software engineering, viết code và chạy test suite.
- **Tool-call Agent** (`toolcall-rl/`): Agent tối ưu hóa việc gọi và sử dụng các công cụ/API.

---

**Giấy phép (License)**: Apache 2.0
