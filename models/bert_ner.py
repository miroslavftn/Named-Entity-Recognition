import torch
from keras.preprocessing.sequence import pad_sequences
from pytorch_pretrained_bert.modeling import BertForTokenClassification
from torch.nn import CrossEntropyLoss
from torch.nn.functional import cross_entropy
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import trange


class Trainer:
    def __init__(self, model, optimizer, epochs, batch_size, device, train_loader, val_loader=None):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.batch_size = batch_size
        self.max_grad_norm = 1.0
        self.device = device
        self.loss_fnc = CrossEntropyLoss()

    def run_train_loop(self):

        for _ in trange(self.epochs, desc="Epoch"):
            self.model.train()

            running_loss = 0.
            total_acc, total_num = 0, 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for i, (input_ids, input_mask, labels) in enumerate(self.train_loader):
                self.model.zero_grad()

                input_ids = input_ids.to(self.device)
                input_mask = input_mask.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                logits = self.model(input_ids=input_ids, token_type_ids=None, attention_mask=input_mask)

                # Compute loss
                loss = cross_entropy(logits.view(-1, self.model.num_labels), labels.view(-1))
                loss.backward()
                # Add mini-batch loss to epoch loss
                running_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                # gradient clipping
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=self.max_grad_norm)
                # Update weights
                self.optimizer.step()
                # Compute accuracy
                preds = logits.argmax(dim=2)
                acc = torch.sum(preds == labels).float()
                total_acc += acc
                total_num += preds.size(0)

            print("Train_loss: {:.4f}".format(running_loss / len(self.train_loader)))
            print("Train_acc: {:.4f}".format(total_acc / total_num))

            # Validation
            if self.val_loader:
                val_acc, val_loss = self.validate()
                print("Val_loss: {:.4f}, Val_acc: {:.4f}".format(val_loss, val_acc))

    def validate(self):
        self.model.eval()  # Set model to eval mode due to Dropout
        running_acc = 0.
        running_loss = 0.
        total_num = 0

        for i, (input_ids, input_mask, labels) in enumerate(self.val_loader):
            # Set all tensors to use cuda
            input_ids = input_ids.to(self.device)
            input_mask = input_mask.to(self.device)
            labels = labels.to(self.device)

            # Make prediction with model
            with torch.no_grad():
                logits = self.model(input_ids=input_ids, token_type_ids=None,
                                    attention_mask=input_mask)

            preds = logits.argmax(dim=2)

            # Compute accuracy
            acc = torch.sum(preds == labels).float()  # / preds.shape[0]
            running_acc += acc
            total_num += preds.size(0)

            # Compute loss
            loss = cross_entropy(logits.view(-1, self.model.num_labels), labels.view(-1))
            running_loss += loss

            # self.model.train()  # Set model back to training mode
        running_loss = running_loss / len(self.val_loader)
        running_acc = running_acc / total_num
        return running_acc, running_loss


def create_data_loader(inputs, tags, masks, batch_size, mode='train'):
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


def tokenize_with_labels(sentences, labels, tokenizer):
    """
    Fix issues caused by BERT tokenizer (word, ###next etc)
    :param sentences:
    :param labels:
    :param tokenizer:
    :return:
    """
    tokens = []
    labels_new = []
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
        labels_new.append(new_tags)
    return tokens, labels_new

def transform_data(tokenizer, sentences, labels, tag2idx, max_len):
    tokenized_texts, labels = tokenize_with_labels(sentences, labels, tokenizer)
    # tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
    input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                              maxlen=max_len, dtype="long", truncating="post", padding="post")
    tags = pad_sequences([[tag2idx.get(l) for l in lab] for lab in labels],
                         maxlen=max_len, value=tag2idx["O"], padding="post",
                         dtype="long", truncating="post")
    attention_masks = [[float(i > 0) for i in ii] for ii in input_ids]

    return input_ids, tags, attention_masks


class BertNER(object):
    def __init__(self, num_labels, train_loader, validation_loader,
                 batch_size=32, epochs=5, device='cuda', full_finetuning=True):
        self.model = None
        self.optimizer = None
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.num_labels = num_labels + 1
        self.epochs = epochs
        self.batch_size = batch_size
        self.full_finetuning = full_finetuning
        self.device = torch.device(device)

        self.init_model()
        self.init_optimizer()
        self.trainer = Trainer(model=self.model,
                               optimizer=self.optimizer,
                               train_loader=self.train_loader,
                               val_loader=self.validation_loader,
                               epochs=self.epochs,
                               batch_size=self.batch_size,
                               device=self.device)

    def init_model(self):
        self.model = BertForTokenClassification.from_pretrained("bert-base-uncased", num_labels=self.num_labels)
        self.model.to(self.device)

    def init_optimizer(self):
        if self.full_finetuning:
            param_optimizer = list(self.model.named_parameters())
            no_decay = ['bias', 'gamma', 'beta']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                 'weight_decay_rate': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                 'weight_decay_rate': 0.0}
            ]
        else:
            param_optimizer = list(self.model.classifier.named_parameters())
            optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]
        self.optimizer = Adam(optimizer_grouped_parameters, lr=3e-5)

    def train(self):
        self.trainer.run_train_loop()
