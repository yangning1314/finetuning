import torch
from torch.optim import AdamW
from transformers import AutoTokenizer,AutoModelForSequenceClassification
from datasets import load_dataset
from transformers import DataCollatorWithPadding
from datasets import load_from_disk
checkpoint = "directory_on_my_computer"
import evaluate

from transformers import AutoModel

model = AutoModel.from_pretrained(checkpoint, num_labels=2)

# checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained("tokenizer")

metric = evaluate.load("data")
# model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
# model.save_pretrained("directory_on_my_computer")

# sequences = [
#     "I've been waiting for a HuggingFace course my whole life.",
#     "This course is amazing!",
# ]
#
# batch = tokenizer(sequences,padding = True,truncation = True, return_tensors="pt")
#
# batch['labels'] = torch.tensor([1,1])
#
# optimizer = AdamW(model.parameters())
#
# loss = model(**batch).loss
# loss.backward()
# optimizer.step()









