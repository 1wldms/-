**3. 추론 방법 (학습 완료 후)**
```python
from unsloth import FastModel
from peft import PeftModel

model, tokenizer = FastModel.from_pretrained(
    "unsloth/gemma-4-E2B-it", load_in_4bit=True
)
model = PeftModel.from_pretrained(model, "lora_model")
FastModel.for_inference(model)

SYSTEM = "시나리오[18]: 요청한 문체/형식/목적에 맞춰 글을 작성하거나 다듬어 주세요."
msgs = [{"role": "user", "content": [{"type": "text", "text": SYSTEM + "\n\n" + your_prompt}]}]
inputs = tokenizer.apply_chat_template(msgs, add_generation_prompt=True,
                                        tokenize=True, return_dict=True,
                                        return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=200, do_sample=True,
                          temperature=0.7, top_p=0.95, top_k=64)
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

### 데이터셋 설계
- 원본 28,978개 → 중복 제거 → 품질 필터링 → 1,000개 샘플링
- 필터링 기준: 응답 150~1,000자, user 입력 20자 이상, 스타일 키워드 포함, 깨진 인코딩 없음
- 학습/검증 분리: train 900 / eval 100

### 하이퍼파라미터
| 손잡이 | 값 | 근거 |
|---|---|---|
| learning rate | 2e-4 | Unsloth 권장 출발값 (wiki/02) |
| epochs | 3 | 과소적합 방지, val loss 모니터링 |
| LoRA r | 16 | 스타일 다양성 고려, r=8 대비 표현력↑ |
| LoRA alpha | 32 | 경험칙 α=2r |
| target modules | All (attn+MLP) | 말투·어휘 동시 학습 필요 |
| effective batch | 16 (2×8) | T4 OOM 방지 |

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
![loss_curve](loss_curve.png)

| Step | Train Loss | Val Loss |
|---|---|---|
| 50 | 0.4150 | 2.7954 |
| 100 | 0.3193 | 2.6708 |
| 150 | 0.3022 | 2.6580 |
| 171 | 0.2980 | 2.6540 |

### Before / After 추론 예시
*(스크린샷 첨부)*

---

## 📚 참고문헌

- Hu et al. (2021). LoRA. arXiv:2106.09685
- Dettmers et al. (2023). QLoRA. arXiv:2305.14314
- Zhou et al. (2023). LIMA. arXiv:2305.11206
- Unsloth Studio: https://unsloth.ai/docs/new/studio
- 데이터셋: https://huggingface.co/datasets/coastral/korean-writing-style-instruct

---

## 📋 재현 정보

| 항목 | 내용 |
|---|---|
| 환경 | Google Colab 무료 / Tesla T4 / 14.563GB |
| 학습 시간 | 약 26분 (1,545초) |
| 랜덤 시드 | 42 |
| 베이스 모델 | unsloth/gemma-4-E2B-it |
| 어댑터 경로 | lora_model/ |
