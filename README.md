# Llama 2 7B — QLoRA Fine-Tuning

A practical implementation of **parameter-efficient fine-tuning (PEFT)** for **Llama 2 7B Chat** using **QLoRA**, 4-bit NF4 quantization, LoRA adapters, and supervised fine-tuning with the Hugging Face ecosystem.

The project demonstrates how a large language model can be adapted to an instruction-following dataset while significantly reducing the memory and compute requirements compared with full-parameter fine-tuning.

---

## 🚀 Project Overview

Large Language Models such as Llama 2 contain billions of parameters, making full fine-tuning expensive and memory-intensive.

This project uses **QLoRA (Quantized Low-Rank Adaptation)** to fine-tune Llama 2 7B efficiently.

Instead of updating the entire model:

```text
Llama 2 7B
   │
   ├── Quantized Base Model (4-bit)
   │       │
   │       └── Frozen
   │
   └── LoRA Adapters
           │
           └── Trainable Parameters
```

The pretrained model is loaded in **4-bit precision**, while lightweight LoRA adapters are trained on top of it.

---

## 🧠 What is QLoRA?

QLoRA combines two techniques:

### Quantization

The pretrained Llama 2 model is loaded using **4-bit quantization**, reducing GPU memory requirements.

This project uses:

* **4-bit quantization:** Enabled
* **Quantization type:** NF4
* **Compute dtype:** FP16
* **Double/Nested Quantization:** Disabled

### LoRA

Instead of updating all parameters of the LLM, LoRA introduces small trainable low-rank matrices into the model.

Conceptually:

```text
Original Weight Matrix
        │
        ├────────────── Frozen
        │
        └── LoRA Adapter
                │
                └── Trainable
```

This makes fine-tuning significantly more parameter-efficient.

---

# 🏗️ Architecture

The overall training pipeline is:

```text
                Guanaco Dataset
                       │
                       ▼
                Dataset Loading
                       │
                       ▼
                 Llama Tokenizer
                       │
                       ▼
              4-bit Quantization
                       │
                       ▼
                Llama 2 7B Chat
                       │
                       ▼
                  LoRA Adapters
                       │
                       ▼
             Supervised Fine-Tuning
                       │
                       ▼
                Fine-Tuned Model
```

---

# 🛠️ Tech Stack

| Technology                | Purpose                                |
| ------------------------- | -------------------------------------- |
| Python                    | Programming language                   |
| PyTorch                   | Deep learning framework                |
| Hugging Face Transformers | LLM loading and training               |
| PEFT                      | LoRA / parameter-efficient fine-tuning |
| TRL                       | Supervised fine-tuning                 |
| BitsAndBytes              | 4-bit quantization                     |
| Hugging Face Datasets     | Dataset loading                        |
| TensorBoard               | Training monitoring                    |
| Accelerate                | Training optimization                  |

---

# 📦 Base Model

The project uses:

**NousResearch/Llama-2-7b-chat-hf**

This is a Llama 2 7B Chat model prepared for use with the Hugging Face Transformers ecosystem.

---

# 📚 Dataset

The instruction dataset used is:

**mlabonne/guanaco-llama2-1k**

The dataset contains approximately **1,000 instruction-following examples** designed for conversational fine-tuning.

Dataset is loaded using:

```python
from datasets import load_dataset

dataset = load_dataset(
    "mlabonne/guanaco-llama2-1k",
    split="train"
)
```

---

# ⚙️ QLoRA Configuration

The LoRA configuration used in the project:

```text
LoRA Rank (r):        64
LoRA Alpha:           16
LoRA Dropout:         0.1
Bias:                 None
Task:                 CAUSAL_LM
```

The relatively small LoRA adapter allows the model to learn task-specific behavior without modifying the complete pretrained model.

---

# 🔢 4-Bit Quantization

The base model is loaded using BitsAndBytes:

```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False
)
```

### Configuration

| Parameter           | Value    |
| ------------------- | -------- |
| 4-bit Loading       | Enabled  |
| Quantization        | NF4      |
| Compute Precision   | FP16     |
| Double Quantization | Disabled |

NF4 is specifically designed for quantizing normally distributed neural network weights and is commonly used in QLoRA workflows.

---

# 🏋️ Training Configuration

The supervised fine-tuning pipeline uses the Hugging Face TRL `SFTTrainer`.

### Training Parameters

```text
Epochs:                  1
Learning Rate:           2e-4
Optimizer:               Paged AdamW 32-bit
Weight Decay:            0.001
LR Scheduler:            Cosine
Warmup Ratio:            0.03
Max Gradient Norm:       0.3
Gradient Checkpointing:  Enabled
```

Gradient checkpointing is enabled to reduce memory consumption during training.

---

# 🔤 Tokenizer Configuration

The Llama tokenizer is loaded directly from the base model.

```python
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
```

The EOS token is used as the padding token to ensure compatibility during training.

---

# 💻 Hardware Considerations

QLoRA is particularly useful when GPU memory is limited.

Instead of loading the entire model in FP16/BF16, the pretrained weights are loaded in 4-bit precision.

```text
Full Fine-Tuning

7B Model
   │
   └── Full Model Parameters Trainable
            │
            ▼
       High GPU Memory


QLoRA

7B Model
   │
   ├── 4-bit Quantized Base Model
   │        └── Frozen
   │
   └── LoRA Adapters
            └── Trainable
                    │
                    ▼
              Lower Memory
```

Actual training requirements depend on sequence length, batch size, GPU architecture, and other runtime settings.

---

# 🔄 Training Workflow

The complete workflow is:

### 1. Install Dependencies

```bash
pip install accelerate peft bitsandbytes transformers trl
```

### 2. Load Dataset

```python
dataset = load_dataset(
    "mlabonne/guanaco-llama2-1k",
    split="train"
)
```

### 3. Configure 4-bit Quantization

```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False
)
```

### 4. Load Llama 2

```python
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map={"": 0}
)
```

### 5. Configure LoRA

```python
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM"
)
```

### 6. Configure Training

The project uses:

```python
TrainingArguments(
    learning_rate=2e-4,
    num_train_epochs=1,
    optim="paged_adamw_32bit",
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    gradient_checkpointing=True
)
```

### 7. Supervised Fine-Tuning

The LoRA configuration is passed into the TRL supervised fine-tuning trainer.

```python
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=False
)
```

### 8. Train

```python
trainer.train()
```

---

# 📊 Training Monitoring

TensorBoard reporting is enabled through:

```python
report_to="tensorboard"
```

Training metrics can therefore be monitored during fine-tuning.

Example workflow:

```bash
tensorboard --logdir ./results
```

This allows monitoring of training behavior and loss throughout the fine-tuning process.

---

# 🎯 Why Parameter-Efficient Fine-Tuning?

Full fine-tuning requires updating billions of parameters.

QLoRA instead keeps the pretrained model largely frozen and learns a small set of LoRA parameters.

### Full Fine-Tuning

```text
7B Parameters
     │
     ▼
All Parameters Updated
     │
     ▼
High Memory + Compute
```

### QLoRA

```text
7B Parameters
     │
     ▼
4-bit Quantized + Frozen
     │
     └─────────────┐
                   ▼
             Small LoRA
               Adapters
                   │
                   ▼
             Trainable
```

This makes experimentation with large language models much more accessible on limited hardware.

---

# 🔬 Key Concepts Demonstrated

This project demonstrates practical understanding of:

* Large Language Models
* Llama 2 architecture
* Causal Language Modeling
* Supervised Fine-Tuning
* Parameter-Efficient Fine-Tuning
* LoRA
* QLoRA
* 4-bit Quantization
* NF4 Quantization
* BitsAndBytes
* Hugging Face Transformers
* Hugging Face PEFT
* Hugging Face TRL
* Gradient Checkpointing
* AdamW Optimization
* Learning Rate Scheduling
* TensorBoard
* GPU Memory Optimization

---

# 📁 Project Structure

```text
FineTune_Llama-2-7b/
│
├── FineTune_Llama_2_7B.ipynb
├── README.md
└── results/
    └── training outputs
```

---

# 🚀 Future Improvements

Possible improvements include:

* [ ] Add evaluation metrics on a held-out dataset
* [ ] Compare base vs fine-tuned model responses
* [ ] Add automated evaluation using LLM-as-a-Judge
* [ ] Experiment with different LoRA ranks
* [ ] Compare QLoRA against standard LoRA
* [ ] Experiment with BF16 on supported GPUs
* [ ] Add model checkpoint saving
* [ ] Merge LoRA adapters with the base model
* [ ] Create an interactive Gradio inference interface
* [ ] Publish the trained adapter to Hugging Face Hub
* [ ] Add experiment tracking with Weights & Biases

---

# 📌 Project Highlights

**Model:** Llama 2 7B Chat
**Fine-Tuning Method:** QLoRA
**Dataset:** Guanaco Llama2 1K
**Quantization:** 4-bit NF4
**LoRA Rank:** 64
**LoRA Alpha:** 16
**Framework:** PyTorch + Hugging Face
**Training:** Supervised Fine-Tuning

---

# 📜 Disclaimer

This repository is intended for **educational and experimental purposes**. Training performance and generated outputs can vary depending on hardware, software versions, dataset preprocessing, and training configuration.

The project demonstrates the mechanics of parameter-efficient LLM fine-tuning rather than production deployment.

---

## 👨‍💻 Author

**Karan Dhariwal**

GitHub:
`https://github.com/karandhariwal`

---

## ⭐ If you found this project useful

Consider giving the repository a ⭐ and exploring the implementation.
