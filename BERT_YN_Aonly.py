import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer, DataCollatorWithPadding
import evaluate
import numpy as np

from huggingface_hub import notebook_login
token = ''
notebook_login()

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def tokenizer_model():
    '''
    This function loads the tokenizer and model
    '''
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_auth_token = token)

    id2label = {0: "Yes", 1: "No", 2: "In the middle, neither yes nor no", 3: "Yes, subject to some conditions)"}
    label2id = {"Yes": 0, "No": 1, "In the middle, neither yes nor no": 2, "Yes, subject to some conditions)": 3}

    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=4, id2label=id2label, label2id=label2id, use_auth_token = token
        )
    return model, tokenizer

model, tokenizer = tokenizer_model()


def dataset_loader():
    '''
    This functions loads the Circa dataset
    and splits it randomly in three parts of
    train, dev, test with 60, 20, 20 percentage
    returns a datasetdict 
    '''
    dataset = load_dataset("circa", split = 'train')

    #filter the unknown data
    dataset = dataset.filter(lambda example: 
                                (example['goldstandard2']==0 or 
                                example['goldstandard2']== 1 or
                                example['goldstandard2']== 2 or 
                                example['goldstandard2']== 3))
    
    train_testvalid = dataset.train_test_split(test_size=0.4, seed=42)
    test_valid = train_testvalid['test'].train_test_split(test_size=0.5, seed=42)
    train_testvalid['test'] = test_valid['test']
    train_testvalid['valid'] = test_valid['train']

    return train_testvalid

dataset = dataset_loader()

def encode(examples):
    '''
    encodes the input with tokenizer
    '''
    return tokenizer(examples['answer-Y'], truncation=True)


def dataset_tokenizer(dataset):
    '''
    This functions tokenizes the dataset and 
    changes the labes of glodstandard2 to labels
    '''
    dataset = dataset.map(encode, batched=True)
    dataset = dataset.map(lambda examples: {"labels": examples["goldstandard2"]}, batched=True)
    return dataset


dataset = dataset_tokenizer(dataset)

#part 5 with PyTorch

dataset.set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "labels"])
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
accuracy = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


#Train part with pytorch
def main():

    training_args = TrainingArguments(
        output_dir="BERT_Answer_only",
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=True,
        hub_token = token
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["valid"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    trainer.push_to_hub()

main()