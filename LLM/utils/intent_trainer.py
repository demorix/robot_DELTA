import pandas as pd
import torch
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)

# =========================
# 📁 1. Upload data
# =========================
df = pd.read_csv("data_for_intent.csv", sep=";", header=None)
df.columns = ["text", "label"]
df["label"] = df["label"].astype(int)

print("Дані:")
print(df.head())
print(df.dtypes)

# =========================
# 🏷️ 2. Labels
# =========================
id2label = {0: "camera", 1: "movement", 2: "speech"}
label2id = {v: k for k, v in id2label.items()}

# =========================
# 🔤 3. Tokeniazer
# =========================
model_path = "./"  

tokenizer = DistilBertTokenizer.from_pretrained(model_path)

encodings = tokenizer(
    df["text"].tolist(),
    truncation=True,
    padding=True
)

# =========================
# 📦 4. Dataset
# =========================
class IntentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

dataset = IntentDataset(encodings, df["label"].tolist())

# =========================
# 🤖 5. Model
# =========================
model = DistilBertForSequenceClassification.from_pretrained(
    model_path,
    num_labels=3,
    id2label=id2label,
    label2id=label2id
)

# =========================
# ⚙️ 6. GPU / CPU
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
model.to(device)

# =========================
# 🚀 7. TrainingArguments
# =========================
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=10,
    per_device_train_batch_size=4,
    logging_steps=1,
    save_strategy="epoch",
    fp16=torch.cuda.is_available(),
    report_to="none"
)

# =========================
# 🏋️ 8. Trainer
# =========================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

# =========================
# ▶️ 9. Run trainer
# =========================
trainer.train()

# =========================
# 💾 10. save llm
# =========================
trainer.save_model("./intent_2model")
tokenizer.save_pretrained("./intent_2model")

print("✅ Model save in ./intent_model")