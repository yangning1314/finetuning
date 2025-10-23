from datasets import load_from_disk
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import  AutoModelForSequenceClassification
# from datasets import load_metric

import numpy as np


from transformers import Trainer

import evaluate


'''
导入数据集
'''


raw_datasets = load_from_disk("data")
checkpoint = "directory_on_my_computer"
tokenizer = AutoTokenizer.from_pretrained("tokenizer")
model =  AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

'''
将特征添加到数据集中
'''
def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer)


'''
用于训练和评估的所有超参数。您必须提供的唯一参数是保存训练模型的目录，以及沿途的检查点。对于所有其他内容，您可以保留默认值
'''
from transformers import TrainingArguments

training_args = TrainingArguments(
    "test-trainer",
    eval_strategy="epoch",
)


def compute_metrics(eval_preds):
    metric = evaluate.load("data")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)



trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
