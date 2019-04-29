import numpy as np
import torch
from keras.preprocessing.sequence import pad_sequences
from pytorch_pretrained_bert.modeling import BertForTokenClassification
from torch.nn import CrossEntropyLoss
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import trange


class Trainer:
    """
    Run training for BERT NER model
    """
    def __init__(self, model: any, optimizer: Adam, epochs: int, batch_size: int, device, num_labels: int,
                 train_loader: DataLoader, val_loader: DataLoader=None):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.batch_size = batch_size
        self.max_grad_norm = 1.0
        self.num_labels = num_labels
        self.device = device
        self.loss_fnc = CrossEntropyLoss()

    def run_train_loop(self):
        for _ in trange(self.epochs, desc="Epoch"):
            self.model.train()
            tr_loss, total_acc = 0, 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(self.train_loader):
                # add batch to gpu
                batch = tuple(t.to(self.device) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch
                # forward pass
                logits = self.model(b_input_ids, token_type_ids=None,
                                    attention_mask=b_input_mask)
                # backward pass
                loss = self.loss_fnc(logits.view(-1, self.num_labels), b_labels.view(-1))
                loss.backward()
                # track train loss
                tr_loss += loss.item()
                nb_tr_examples += b_input_ids.size(0)
                nb_tr_steps += 1
                # gradient clipping
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=self.max_grad_norm)
                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()
                total_acc += flat_accuracy(logits, label_ids)
                # update parameters
                self.optimizer.step()
                self.model.zero_grad()

            print("Train_loss: {:.4f}".format(tr_loss / nb_tr_steps))
            print("Train_acc: {:.4f}".format(total_acc / nb_tr_steps))

            # Validation
            if self.val_loader:
                val_acc, val_loss = self.validate()

    def validate(self):
        self.model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        for batch in self.val_loader:
            batch = tuple(t.to(self.device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch

            with torch.no_grad():
                logits = self.model(b_input_ids, token_type_ids=None,
                                    attention_mask=b_input_mask)
                tmp_eval_loss = self.loss_fnc(logits.view(-1, self.num_labels), b_labels.view(-1))
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            tmp_eval_accuracy = flat_accuracy(logits, label_ids)

            eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy

            nb_eval_examples += b_input_ids.size(0)
            nb_eval_steps += 1
        eval_loss = eval_loss / nb_eval_steps
        eval_acc = eval_accuracy / nb_eval_steps
        print("Validation loss: {}".format(eval_loss))
        print("Validation Accuracy: {}".format(eval_acc))

        return eval_acc, eval_loss


def create_data_loader(inputs, tags, masks, batch_size:int, mode: str = 'train')-> DataLoader :
    """
    Create data loader from inputs, tags and masks
    :param inputs:
    :param tags:
    :param masks:
    :param batch_size:
    :param mode: train and val
    :return:
    """
    inputs = torch.tensor(inputs)
    tags = torch.tensor(tags)
    masks = torch.tensor(masks)

    data = TensorDataset(inputs, masks, tags)
    if mode == 'train':
        sampler = RandomSampler(data)
    else:
        sampler = SequentialSampler(data)
    data_loader = DataLoader(data, sampler=sampler, batch_size=batch_size)
    return data_loader


def fix_tokenizer(sentences, labels, tokenizer):
    tokens = []
    new_labels = []
    sentences_clean = [sent.lower() for sent in sentences]
    for sent, tags in zip(sentences_clean, labels):
        new_tags = []
        new_text = []
        for word, tag in zip(sent.split(), tags):
            sub_words = tokenizer.tokenize(word)
            for count, sub_word in enumerate(sub_words):
                if count > 0:
                    tag = 'X'
                new_tags.append(tag)
                new_text.append(sub_word)
        tokens.append(new_text)
        new_labels.append(new_tags)
    return tokens, new_labels


def transform_data(tokenizer, sentences, labels, tag2idx, max_len):
    # tokenized_texts, labels = fix_tokenizer(sentences, labels, tokenizer)
    tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
    input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                              maxlen=max_len, dtype="long", truncating="post", padding="post")
    tags = pad_sequences([[tag2idx.get(l) for l in lab] for lab in labels],
                         maxlen=max_len, value=tag2idx["O"], padding="post",
                         dtype="long", truncating="post")
    attention_masks = [[float(i > 0) for i in ii] for ii in input_ids]

    return input_ids, tags, attention_masks


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=2).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


class BertNER(object):
    def __init__(self, num_labels, train_loader, validation_loader,
                 batch_size=32, epochs=5, device='cuda', learning_rate=2e-5):
        self.model = None
        self.optimizer = None
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.num_labels = num_labels
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = torch.device(device)
        # self.num_train_steps = int(len(self.train_loader)) * self.epochs
        self.max_grad_norm = 1.0
        self.init_model()
        self.init_optimizer()

        self.trainer = Trainer(model=self.model,
                               optimizer=self.optimizer,
                               train_loader=self.train_loader,
                               val_loader=self.validation_loader,
                               epochs=self.epochs,
                               batch_size=self.batch_size,
                               num_labels=self.num_labels,
                               device=self.device)

    def init_model(self):
        self.model = BertForTokenClassification.from_pretrained("bert-base-uncased", num_labels=self.num_labels)
        self.model.to(self.device)

    def init_optimizer(self):
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        self.optimizer = Adam(optimizer_grouped_parameters,
                              lr=self.learning_rate)

    def train(self):
        self.trainer.run_train_loop()
