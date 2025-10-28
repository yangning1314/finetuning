from datasets import load_from_disk
from transformers import AutoTokenizer, DataCollatorWithPadding, EarlyStoppingCallback
from transformers import  AutoModelForSequenceClassification
# from datasets import load_metric

# Example of tracking loss during training with the Trainer
from transformers import Trainer, TrainingArguments
import wandb

import numpy as np

import evaluate


'''
导入数据集
AutoModelForSequenceClassification不能换AutoModel，因为AutoModel中会产生label参数与labels不兼容
'''


raw_datasets = load_from_disk("data")
checkpoint = "directory_on_my_computer"
tokenizer = AutoTokenizer.from_pretrained("datasets_on_my_computer")
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


'''
evaluate用来评估
evaluate.load()用来加载评估器，线上数据集自带评估器，但是本地数据集要将GitHub上的evaluate clone下来，
在函数中传入要使用的评估器的路径
predictions用来预测
'''
def compute_metrics(eval_preds):
    metric = evaluate.load("data")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# Initialize Weights & Biases for experiment tracking
# wandb.init(project="transformer-fine-tuning", name="bert-mrpc-analysis")

'''
TrainerAPI用来训练模型
'''


training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="steps",
    eval_steps=50,
    save_steps=100,
    logging_steps=10,  # Log metrics every 10 steps
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    # report_to="wandb",  # Send logs to Weights & Biases
)

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)


#loss曲线
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_datasets["train"],
#     eval_dataset=tokenized_datasets["validation"],
#     data_collator=data_collator,
#     processing_class=tokenizer,
#     compute_metrics=compute_metrics,
# )

#过拟合
# training_args = TrainingArguments(
#     output_dir="./results",
#     eval_strategy="steps",
#     eval_steps=100,
#     save_strategy="steps",
#     save_steps=100,
#     load_best_model_at_end=True,
#     metric_for_best_model="eval_loss",
#     greater_is_better=False,
#     num_train_epochs=10,  # Set high, but we'll stop early
# )
#
# # Add early stopping to prevent overfitting
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_datasets["train"],
#     eval_dataset=tokenized_datasets["validation"],
#     data_collator=data_collator,
#     processing_class=tokenizer,
#     compute_metrics=compute_metrics,
#     callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
# )

#欠拟合
# training_args = TrainingArguments(
#     output_dir="./results",
#     eval_strategy="steps",
#     eval_steps=100,
#     save_strategy="steps",
#     save_steps=100,
#     load_best_model_at_end=True,
#     metric_for_best_model="eval_loss",
#     greater_is_better=False,
#     -num_train_epochs=5,
#     +num_train_epochs=10,
# # Set high, but we'll stop early
# )
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_datasets["train"],
#     eval_dataset=tokenized_datasets["validation"],
#     data_collator=data_collator,
#     processing_class=tokenizer,
#     compute_metrics=compute_metrics,
#     callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
# )


#不稳定的学习曲线
# training_args = TrainingArguments(
#     output_dir="./results",
#     eval_strategy="steps",
#     eval_steps=50,
#     save_steps=100,
#     logging_steps=10,  # Log metrics every 10 steps
#     num_train_epochs=3,
#     -learning_rate=1e-5,
#     +learning_rate=1e-4,
#     -per_device_train_batch_size=16,
#     +per_device_train_batch_size=32,
#     report_to="wandb",
# )
# training_args = TrainingArguments(
#     output_dir="./results",
#     eval_strategy="steps",
#     eval_steps=50,
#     save_steps=100,
#     logging_steps=10,  # Log metrics every 10 steps
#     num_train_epochs=3,
#     per_device_train_batch_size=16,
#     per_device_eval_batch_size=16,
#     report_to="wandb",  # Send logs to Weights & Biases
# )
#
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_datasets["train"],
#     eval_dataset=tokenized_datasets["validation"],
#     data_collator=data_collator,
#     processing_class=tokenizer,
#     compute_metrics=compute_metrics,
#     callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
# )

trainer.train()
