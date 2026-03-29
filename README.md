# 🚜 KrishiBooking AI
### *Distilling SOTA Intelligence into an Edge-Ready Agricultural Booking Assistant*

> **ML Kolkata Hackathon 2026** · Built with Smolify · Fine-tuned on Gemma 3 (270M) · Runs on CPU

---

## 🌾 The Problem

India has **150+ million smallholder farmers**. Most of them:

- Speak in **mixed, informal language** — Hindi, Bengali, English blended together
- Have **no access to formal booking apps** that understand them
- Depend on **expensive middlemen** to rent agricultural equipment like tractors
- Live in areas with **poor or no internet connectivity**

Existing solutions like GPT-4 or Gemini are:
- Too expensive per query for rural deployments
- Require constant internet connectivity
- Not optimized for Indic informal dialects
- Overkill — a farmer just needs a tractor booked, not a general chatbot

**There was no lightweight, offline-capable, multilingual model that could do one job perfectly: understand a farmer's request and convert it into a structured tractor booking.**

---

## 💡 The Solution — KrishiBooking AI

KrishiBooking AI is a **Domain Specific Language Model (DSLM)** fine-tuned on Gemma 3 (270M) that converts natural language farmer requests into structured JSON booking objects.

It understands:
- 🇮🇳 **Hindi** — *"Kal tractor chahiye 3 ghante ke liye hal chalane ke liye"*
- 🇧🇩 **Bengali** — *"Aj boro jomi te tractor dorkar ekhuni, 4 ghanta"*
- 🇬🇧 **English** — *"I need a tractor for harvesting tomorrow in Hooghly"*
- 🔀 **Mixed/Informal** — *"Bhai ek tractor bhej de aj jaldi, Meerut mein kaam hai"*

And outputs clean, structured JSON every single time:

```json
{
  "service": "tractor",
  "task": "ploughing",
  "date": "tomorrow",
  "duration_hours": 3,
  "location": "Hooghly",
  "urgency": "high"
}
```

---

## 🧠 Why a Specialized Model? Not a General LLM?

This is the core architectural decision of this project.

| Factor | General LLM (GPT-4/Gemini) | KrishiBooking AI (Gemma 270M) |
|---|---|---|
| **Cost per query** | ~$0.01–$0.03 | $0.00 (runs locally) |
| **Internet required** | Yes, always | No — fully offline |
| **Latency** | 1–5 seconds (network) | <200ms (on-device) |
| **Privacy** | Data leaves device | Data stays local |
| **Model size** | 100B+ parameters | 270M parameters |
| **Task accuracy** | Good (generalist) | Higher (specialized) |
| **Deployment** | Cloud only | Android, Raspberry Pi, CPU |

A general LLM is like hiring a surgeon to book your cab. KrishiBooking AI is the cab booking app — **small, fast, and purpose-built**.

By applying **Knowledge Distillation**, we transferred only the relevant reasoning capability from a large teacher model (Gemini) into a tiny student model (Gemma 270M). The student doesn't need to know about history, poetry, or code — it only needs to understand one thing: what a farmer is asking for.

This is the **DeepSeek Effect** in action — specialized efficiency beats generalist scale for domain problems.

---

## 🏗️ Architecture & Pipeline

```
Farmer Input (Hindi/Bengali/English/Mixed)
        │
        ▼
┌─────────────────────────────────────┐
│         KrishiBooking AI            │
│      Gemma 3 (270M) Fine-tuned      │
│   via Smolify Knowledge Distillation│
└─────────────────────────────────────┘
        │
        ▼
Structured JSON Booking Object
        │
        ▼
Backend Booking System / WhatsApp Bot / Government Portal
```

### Distillation Pipeline (via Smolify)

1. **Define Mission** — System persona, task description, multilingual examples
2. **Teacher Synthesis** — Gemini 2.5 Flash generates ~10,000 high-fidelity training pairs
3. **Data Review** — Synthetic examples verified for accuracy across all languages
4. **Fine-tuning** — Gemma 3 (270M) trained via LoRA/QLoRA on L4 GPU
5. **Edge Deployment** — Model exported to Hugging Face, runs on CPU

---

## 📊 Example Inputs & Outputs

### Hindi Input
```
Input:  "Mujhe hal chalwane ke liye tractor chahiye Punjab mein, kal 5 ghante ke liye"
Output: {"service":"tractor","task":"ploughing","date":"tomorrow","duration_hours":5,"location":"Punjab","urgency":"medium"}
```

### Bengali Input
```
Input:  "Ekhuni ektau tractor chai harvesting korar jonno Bankura te, 3 ghanta lagbe"
Output: {"service":"tractor","task":"harvesting","date":"today","duration_hours":3,"location":"Bankura","urgency":"high"}
```

### Mixed/Informal Input
```
Input:  "Bhai ek tractor bhej de aj jaldi hi, 2 ghante ka kaam hai Meerut mein"
Output: {"service":"tractor","task":"field_work","date":"today","duration_hours":2,"location":"Meerut","urgency":"high"}
```

### English Input
```
Input:  "Please book tractor for tomorrow for field work at Ranchi"
Output: {"service":"tractor","task":"field_work","date":"tomorrow","duration_hours":null,"location":"Ranchi","urgency":"medium"}
```

---

## 🚀 Run Locally on CPU

```python
from transformers import AutoProcessor, AutoModelForCausalLM

model_id = "your-hf-username/krishibooking-ai"

processor = AutoProcessor.from_pretrained(model_id, device_map="auto")
model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto", device_map="auto")

message = [
    {"role": "system", "content": "You are KrishiBooking AI. Convert farmer requests to JSON booking objects only."},
    {"role": "user", "content": "Kal tractor chahiye 3 ghante ke liye hal chalane ke liye Hooghly mein"}
]

inputs = processor.apply_chat_template(message, add_generation_prompt=True, return_dict=True, return_tensors="pt")
out = model.generate(**inputs.to(model.device), pad_token_id=processor.eos_token_id, max_new_tokens=128)
output = processor.decode(out[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)

print(output)
# {"service":"tractor","task":"ploughing","date":"tomorrow","duration_hours":3,"location":"Hooghly","urgency":"medium"}
```

---

## 🎯 Real-World Impact

- **Target Users**: 150M+ smallholder farmers across India
- **Deployment Target**: WhatsApp bots, Android devices, government agri-portals
- **Languages Supported**: Hindi, Bengali, English, mixed/informal dialects
- **Hardware Requirement**: Any CPU — no GPU, no cloud, no internet needed
- **Cost at Scale**: ₹0 per query after deployment

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Base Model | Google Gemma 3 (270M) |
| Distillation Platform | Smolify (smolify.ai) |
| Teacher Model | Gemini 2.5 Flash |
| Fine-tuning Method | LoRA / QLoRA |
| Inference | HuggingFace Transformers |
| Model Hosting | Hugging Face Hub |
| Languages | Python |

---

## 🤗 Hugging Face Links

**Dataset:** https://huggingface.co/datasets/rjavi1/smolified-krishibooking-ai

**Model:** https://huggingface.co/rjavi1/smolified-krishibooking-ai


## 👤 About

Built at the **ML Kolkata Hackathon — March 29, 2026**

Hosted by Rishiraj Acharya (Google Developer Expert, Founder at Smolify) as part of the *"Build with Gemma locally on CPU by Distilling SOTA Intelligence"* mini-hackathon.

---

*KrishiBooking AI — Because every farmer deserves technology that speaks their language.*
