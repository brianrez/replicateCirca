from transformers import AutoTokenizer
import torch
from transformers import AutoModelForSequenceClassification
from datasets import load_dataset


def classifier(question, answer):

    tokenizer = AutoTokenizer.from_pretrained("mhr2004/BERT_QandA")
    inputs = tokenizer(question, answer, return_tensors="pt")
    model = AutoModelForSequenceClassification.from_pretrained("mhr2004/BERT_QandA")
    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_class_id = logits.argmax().item()

    return model.config.id2label[predicted_class_id]

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


id2label = {0: "Yes", 1: "No", 2: "In the middle, neither yes nor no", 3: "Yes, subject to some conditions)"}

all = 0
correct = 0
for line in dataset['test']:
    if classifier(line['question-X'], line['answer-Y']) == id2label[line['goldstandard2']]:
        correct += 1
        all += 1
    else:
        all +=1 

print('accuracy equals to: ' + str(correct / all))
