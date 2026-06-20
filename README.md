# 🇰🇷 한국어 글쓰기 스타일 변환
> Scenario 18 — 예시 문장을 보고 해당 문체로 새로운 내용을 생성하는 정책을 SFT로 주입한 파인튜닝 프로젝트

---

## 📌 프로젝트 소개

예시 문장과 작성할 내용을 함께 입력하면, 예시 문장의 말투·어휘·문장 구조를 유지하면서 새로운 글을 생성하는 모델입니다.
기존 모델은 예시 문장의 말투를 무시하고 마크다운·정형화된 형식을 따랐습니다. 이는 스타일 지식이 없는 게 아니라 **정책 결함** 문제로, SFT를 통해 교정했습니다.

### 입출력 예시

**입력:**
```
문체 예시: "갑자기 TMI인데 어제 꿀잼 드라마 정주행했음ㅋㅋ"
작성할 내용: 새로 나온 드라마에 대해 온라인 커뮤니티에서 의견을 나누는 글
```
**Before (베이스 모델):**
```
## 🎬 [드라마 제목] 정주행 후기!
**좋았던 점:**
1. 스토리 전개: ...
```
**After (파인튜닝 모델):**
```
ㅇㅇㅇ님들 요즘 뭐 보냐?? ㅋㅋㅋ
나 어제 완전 꽂힌 드라마 있는데 완전 꿀잼임ㅠㅠㅠ
갓띵작 각임ㅇㅇ 무조건 정주행 ㄱㄱ
```
---

## ⚙️ 설치 방법 / 실행 방법
### Google Colab (권장, T4 GPU)

**1. 환경 설치**

```bash
!pip install -q "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install -q --no-deps trl peft accelerate bitsandbytes
!pip install -q datasets
```

**2. 노트북 실행**
`자연어_기말_프로젝트.ipynb`를 Colab에서 열고 아래 순서대로 실행합니다.

```
① 모델 로드
② Before 실험 (베이스 모델 출력 확인)
③ LoRA 어댑터 부착
④ 데이터 준비 (필터링 → 샘플링 → 분리 → chat template)
⑤ SFTTrainer 학습
⑥ Export (LoRA 어댑터 저장)
⑦ Loss 곡선 시각화
⑧ 망각 체크
⑨ After 실험 (파인튜닝 모델 출력 확인)
```

**3. 추론 방법 (학습 완료 후)**
```python
from unsloth import FastModel
from peft import PeftModel

model, tokenizer = FastModel.from_pretrained(
    "unsloth/gemma-4-E2B-it",
    load_in_4bit=True
)
model = PeftModel.from_pretrained(model, "lora_model")
FastModel.for_inference(model)

SYSTEM = "시나리오[18]: 요청한 문체/형식/목적에 맞춰 글을 작성하거나 다듬어 주세요."
prompt = '예시 문장: "갑자기 TMI인데 어제 꿀잼 드라마 정주행했음ㅋㅋ"\n작성할 내용: 새로 나온 드라마 후기'

msgs = [{"role": "user", "content": [{"type": "text", "text": SYSTEM + "\n\n" + prompt}]}]
inputs = tokenizer.apply_chat_template(
    msgs, add_generation_prompt=True,
    tokenize=True, return_dict=True, return_tensors="pt"
).to("cuda")

outputs = model.generate(
    **inputs, max_new_tokens=200,
    do_sample=True, temperature=0.7, top_p=0.95, top_k=64
)
print(tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True))
```

---

## ✨ 주요 기능
파인튜닝 모델이 베이스 모델보다 잘하는 것:

- 예시 문장의 말투·어휘·문체를 따라가는 글 생성
- 마크다운·대본 형식 없이 자연스러운 구어체 출력
- 할머니 말투, 인터넷 슬랭 등 다양한 한국어 스타일 변환

---

## 🛠️ 기술 스택
| 항목 | 내용 |
|---|---|
| 베이스 모델 | unsloth/gemma-4-E2B-it (2B, instruct) |
| 파인튜닝 방법 | QLoRA (load_in_4bit=True, r=16, alpha=32) |
| 학습 프레임워크 | Unsloth + TRL SFTTrainer |
| 데이터셋 | coastral/korean-writing-style-instruct |
| 실행 환경 | Google Colab 무료 (T4 15GB) |

---

## 📐 파인튜닝 설계
### 모델 선택 근거
- **E2B (2B)**: 4bit QLoRA 적용 시 T4 15GB에서 여유 있게 학습 가능
- **instruct(-it)**: 대화 포맷 이미 학습 → 스타일 정책만 추가 주입
- **gemma-4**: 다국어 사전학습 포함 → 한국어 능력 충분
- **Unsloth 공식 지원**: FastModel.from_pretrained()로 바로 로드 가능

### 데이터셋 설계
- 원본 28,978개 → 중복 제거 → 품질 필터링 → 1,000개 샘플링
- 필터링 기준: 응답 150~1,000자, user 입력 20자 이상, 스타일 키워드 포함, 깨진 인코딩 없음
- 학습/검증 분리: train 900 / eval 100

### 하이퍼파라미터 설정
| 손잡이 | 값 | 근거 |
|---|---|---|
| learning rate | 2e-4 | Unsloth 권장 출발값 (wiki/02) |
| epochs | 3 | 과소적합 방지, val loss 모니터링 |
| LoRA r | 16 | 스타일 다양성 고려, r=8 대비 표현력↑ |
| LoRA alpha | 32 | 경험칙 α=2r (wiki/02) |
| target modules | All (attn+MLP) | 말투·어휘 동시 학습 필요 |
| effective batch | 16 (batch 2 × accum 8) | T4 OOM 방지 |

---

## 📊 베이스 vs 파인튜닝 비교 결과
| 프롬프트 | 베이스 출력 | 파인튜닝 출력 | 평가 |
|---|---|---|---|
| 할머니 말투 예시 + 옛날이야기 작성 | 마크다운 대본 형식, 구어체 미반영 | "할미가 옛날이야기 하나 해줄까? 옛날 옛날에 말이야..." | △ 부분 개선 |
| 인터넷 슬랭 예시 + 드라마 후기 | 마크다운 헤더, 정형화된 게시글 | "ㅇㅇㅇ님들 요즘 뭐 보냐?? ㅋㅋㅋ 완전 꿀잼임ㅠㅠㅠ" | ✅ 개선 |
| 망각 체크 — 수도 | 세종(오답) | 서울(정답) | ✅ 유지 이상 |
| 망각 체크 — 번역 | 자연스러운 번역 | 자연스러운 번역 | ✅ 유지 |

---

## 🖼️ 실행 화면
### Loss 곡선
<img width="790" height="390" alt="image" src="https://github.com/user-attachments/assets/d34ad957-a24d-40a0-9e8d-616672fbc760" />

| Step | Train Loss | Val Loss |
|---|---|---|
| 50 | 0.4150 | 2.7954 |
| 100 | 0.3193 | 2.6708 |
| 150 | 0.3022 | 2.6580 |
| 171 | 0.2980 | 2.6540 |

### Before / After 추론 예시

<img width="840" height="467" alt="Screenshot 2026-06-20 at 22 55 53" src="https://github.com/user-attachments/assets/80b6113d-0ca8-4ed1-8fbf-c1721ac6a4f2" />
<img width="690" height="456" alt="Screenshot 2026-06-20 at 22 56 17" src="https://github.com/user-attachments/assets/c5e11126-5633-4a30-b383-732c78ac5589" />


---

## 📚 참고문헌

- Hu et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. arXiv:2106.09685. https://arxiv.org/abs/2106.09685
- Dettmers et al. (2023). QLoRA: Efficient Finetuning of Quantized LLMs. arXiv:2305.14314. https://arxiv.org/abs/2305.14314
- Zhou et al. (2023). LIMA: Less Is More for Alignment. arXiv:2305.11206. https://arxiv.org/abs/2305.11206
- Unsloth Studio 문서: https://unsloth.ai/docs/new/studio
- Unsloth Fine-tuning Guide: https://unsloth.ai/docs/get-started/fine-tuning-llms-guide
- Google Gemma 4 모델 카드: https://huggingface.co/google/gemma-4-2b-it
- coastral/korean-writing-style-instruct: https://huggingface.co/datasets/coastral/korean-writing-style-instruct

---

## 📋 재현 정보

| 항목 | 내용 |
|---|---|
| 환경 | Google Colab 무료 / Tesla T4 / 14.563GB |
| 학습 시간 | 약 26분 (1,545초) |
| 랜덤 시드 | 42 |
| 베이스 모델 | unsloth/gemma-4-E2B-it |
| 학습 샘플 수 | train 900 / eval 100 |
| 에폭 수 | 3 |
| 어댑터 저장 경로 | lora_model/ |
