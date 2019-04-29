from pytorch_pretrained_bert import BertTokenizer
from sklearn.model_selection import train_test_split

from data_processing.sentence_getter import SentenceGetter
from models.bert_ner import BertNER
from models.bert_ner import transform_data, create_data_loader
MAX_LEN = 64
BATCH_SIZE = 32


def main():
    getter = SentenceGetter(file_path='../data/ner_dataset.csv')
    data = getter.data
    sentences = [" ".join([s[0] for s in sent]) for sent in getter.sentences]
    labels = [[s[2] for s in sent] for sent in getter.sentences]
    tags_vals = list(set(data["Tag"].values))
    tag2idx = {t: i for i, t in enumerate(tags_vals)}

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    input_ids, tags, attention_masks = transform_data(tokenizer, sentences, labels, tag2idx, max_len=MAX_LEN)
    tr_inputs, val_inputs, tr_tags, val_tags = train_test_split(input_ids, tags,
                                                                random_state=2018, test_size=0.1)
    tr_masks, val_masks, _, _ = train_test_split(attention_masks, input_ids,
                                                 random_state=2018, test_size=0.1)

    train_dataloader = create_data_loader(tr_inputs, tr_tags, tr_masks, batch_size=BATCH_SIZE, mode='train')
    valid_dataloader = create_data_loader(val_inputs, val_tags, val_masks, batch_size=BATCH_SIZE, mode='val')

    model = BertNER(num_labels=len(tag2idx),
                    train_loader=train_dataloader,
                    validation_loader=valid_dataloader,
                    batch_size=BATCH_SIZE,
                    epochs=5)
    model.train()


if __name__ == '__main__':
    main()