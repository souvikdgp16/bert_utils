import os
import argparse
import pandas as pd
import numpy as np
import torch, torch.nn
import transformers
from transformers import BertModel, BertTokenizer, BertConfig
from transformers import RobertaModel, RobertaTokenizer, RobertaConfig
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score,confusion_matrix
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler


os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

#Constants

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
n_layers=4


def transformer_preprocess(data, tokenizer):
    input_ids = []
    attention_masks = []
    input_types = []
    for i in data:
        tokenizer_op = tokenizer.encode_plus(i, 
                              max_length=MAX_LEN,
                              pad_to_max_length = True,
                              truncation=True,
                              return_token_type_ids = True,
                              return_attention_mask = True)
        input_ids.append(tokenizer_op['input_ids'])
        attention_masks.append(tokenizer_op['attention_mask'])
        input_types.append(tokenizer_op['token_type_ids'])
    return torch.tensor(input_ids), torch.tensor(attention_masks), torch.tensor(input_types)


def load_model(model_path):
    model = SequenceClassifier(transformer_model, config, n_layers)
    model.load_state_dict(torch.load('{model_path}'.format(model_path=model_path)))
    model.eval()

    return model


def predict(text, model):
    tr_class = torch.tensor([0]).long()
    train_input_ids, train_attention_masks, train_input_types = transformer_preprocess([text], tokenizer)
    train_data = TensorDataset(train_input_ids, train_attention_masks, train_input_types, tr_class)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=1)

    for batch in train_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_segment, y_true = batch

        with torch.no_grad():
            outputs = model(b_input_ids, b_input_mask, b_segment)

        classification_logits = outputs[0].detach().cpu().numpy()
        class_assigned = list(np.argmax(classification_logits,axis=1))
        confidence = F.softmax(torch.tensor(classification_logits), dim=1).cpu().detach().numpy()[0][class_assigned[0]]

    return class_assigned, confidence


class SequenceClassifier(torch.nn.Module):
    def __init__(self, transformer_model, config, n_layers=4, num_classes=2):
        super().__init__()
        self.num_classes = num_classes
        self.transformer = transformer_model
        self.out = torch.nn.Linear(config.hidden_size*2, self.num_classes)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.n_layers = n_layers
  
    def forward(self, input_ids, attention_mask, segment_id, classification_labels=None):
    
        #Batch max length
        max_length = (attention_mask != 0).max(0)[0].nonzero()[-1].item()+1
        if max_length < input_ids.shape[1]:
          input_ids = input_ids[:, :max_length]
          attention_mask = attention_mask[:, :max_length]

        segment_id = torch.zeros_like(attention_mask)
        hidden = self.transformer(input_ids = input_ids,attention_mask = attention_mask, 
                                  token_type_ids = segment_id)
        
        token_hidden = hidden[2][-self.n_layers:]
        token_hidden = torch.mean(torch.sum(torch.stack(token_hidden), dim=0), 
                                  dim=1)

        classifier_hidden = hidden[1]
        hidden_cat = token_hidden
        hidden_cat = torch.cat([token_hidden, classifier_hidden], dim=1)

        classification_logits = self.out(self.dropout(hidden_cat))
        outputs = [classification_logits]
        if classification_labels is not None:
            loss_fct_classification = torch.nn.CrossEntropyLoss()

            loss_classification = loss_fct_classification(classification_logits.view(-1, self.num_classes), 
                                                          classification_labels.view(-1))

            outputs += [loss_classification]
        return outputs


def main():
    max_grad_norm = 1.0
    best_f1 = 0

    parser = argparse.ArgumentParser()
    parser.add_argument("--max_length", type=int, default=386)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--test_size", type=float, default=0.3)
    parser.add_argument("--model_name_save", type=str)
    parser.add_argument("--data_file", type=str)

    args = parser.parse_args()

    try:
        MAX_LEN = args['max_length']
        bs = args['batch_size']
        epochs = args['epochs']
        model_name_save=args['model_name_save']
        data_file=args['data_file']
        num_classes=args['num_classes']
        test_size=args['test_size']
    except:
        print("ERROR in parsing tokens!")

    data=pd.read_csv(data_file)
    X= data["text"].to_list()
    y=data["label"].to_list()

    x_train, x_dev, y_train, y_dev = train_test_split(X, y, test_size=test_size, random_state=42)

    train_input_ids, train_attention_masks, train_input_types = transformer_preprocess(x_train, tokenizer)
    print(train_input_ids.shape, train_attention_masks.shape, train_input_types.shape)

    dev_input_ids, dev_attention_masks, dev_input_types = transformer_preprocess(x_dev, tokenizer)
    print(dev_input_ids.shape, dev_attention_masks.shape, dev_input_types.shape)

    tr_class = torch.tensor(y_train).long()
    val_class = torch.tensor(y_dev).long()

    train_data = TensorDataset(train_input_ids, train_attention_masks, 
                           train_input_types, tr_class)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=bs)

    valid_data = TensorDataset(dev_input_ids, dev_attention_masks, 
                               dev_input_types, val_class)
    valid_sampler = RandomSampler(valid_data)
    valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=bs)

    test_sampler = SequentialSampler(valid_data)
    test_dataloader = DataLoader(valid_data, sampler=test_sampler, batch_size=bs)


    if model_name == 'bert-base-uncased':
        tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=True)
        config = BertConfig.from_pretrained(model_name)
    else:
        tokenizer = RobertaTokenizer.from_pretrained(model_name, do_lower_case=True)
        config = RobertaConfig.from_pretrained(model_name)

        
    config.output_hidden_states = True

    model = SequenceClassifier(transformer_model=transformer_model, config=config, n_layers=n_layers, num_classes=num_classes)
    model.zero_grad()

    FULL_FINETUNING = True
    if FULL_FINETUNING:
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]
    else:
        param_optimizer = list(model.classifier.named_parameters()) 
        optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]
    optimizer = Adam(optimizer_grouped_parameters, lr=2e-5)


    for _ in tqdm(range(epochs)):
    # TRAIN loop
        model.train()
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for batch in tqdm(train_dataloader):

            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_segment, b_classes = batch

            outputs = model(b_input_ids, b_input_mask, b_segment, b_classes)

            loss = outputs[1]

            loss.backward()

            tr_loss += loss.item()
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1

            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), 
                                         max_norm=max_grad_norm)

            optimizer.step()
            # scheduler.step()
            model.zero_grad()

        print("Train loss: {}".format(tr_loss/nb_tr_steps))

        # VALIDATION on validation set
        model.eval()
        eval_loss, eval_accuracy_class = 0, 0
        
        nb_eval_steps, nb_eval_examples = 0, 0
        class_preds ,class_true_labels = [], []
        

        for batch in valid_dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_segment, b_classes = batch

            with torch.no_grad():
                outputs = model(b_input_ids, b_input_mask, b_segment, b_classes)
                eval_loss = classification_loss = outputs[1]
                classification_logits = outputs[0].detach().cpu().numpy()
                class_preds.extend(list(np.argmax(classification_logits,axis=1)))
                class_label_ids = b_classes.to('cpu').numpy()
                class_true_labels.append(class_label_ids)
                tmp_eval_accuracy_class = flat_accuracy_classification(classification_logits, 
                                                      class_label_ids)
                eval_loss += eval_loss.mean().item()
                eval_accuracy_class += tmp_eval_accuracy_class
                nb_eval_examples += b_input_ids.size(0)
                nb_eval_steps += 1
        eval_loss = eval_loss/nb_eval_steps

        print("Validation loss: {}".format(eval_loss))
        print("Validation Accuracy Classifier: {}".format(eval_accuracy_class/nb_eval_steps))
        valid_tags = [l_i for l in class_true_labels for l_i in l ]
        pred_tags = [p for p in class_preds]
        print("Val F1 ",f1_score(valid_tags, pred_tags, average='micro'))
        print("Val Precision ",precision_score(valid_tags, pred_tags, average='micro'))
        print("Val Recall ",recall_score(valid_tags, pred_tags, average='micro'))
        print("Val Accuracy ",accuracy_score(valid_tags, pred_tags))
        print("Confusion matrix \n", confusion_matrix(valid_tags, pred_tags))
        if f1_score(valid_tags, pred_tags, average='micro') >= best_f1:
            print("Saving this model")
            torch.save(model.state_dict(), './{model_name_save}'.format(model_name_save=model_name_save))






