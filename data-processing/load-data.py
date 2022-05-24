import torch
from torch.utils.data import DataLoader
from datasets import load_dataset


# Pile Dataset
def encode(data):
    return tokenizer(data['text'], padding='max_length', truncation=True, max_length=512)


def pile_dataloader(tokenizer, train_path):
    all_data = load_dataset("csv",
                            data_files={
                                "train": train_path,
                                "val": 'data/pile_validation.csv',
                                'test': 'data/pile_test.csv'})

    all_data = all_data.map(encode)

    all_data.set_format(type="torch",
                        columns=['input_ids', 'attention_mask'])

    train_dataloader = DataLoader(all_data['train'],
                                  batch_size=8,
                                  shuffle=True,
                                  num_workers=1)
    val_dataloader = DataLoader(all_data['val'],
                                batch_size=8,
                                num_workers=1)
    test_dataloader = DataLoader(all_data['test'],
                                 batch_size=8,
                                 num_workers=1)

    return train_dataloader, val_dataloader, test_dataloader

# Penn Treebank dataset
def ptb_encode(data):
    return tokenizer(data['sentence'])


def ptb_group_texts(texts):
    concatenated_examples = {k: sum(texts[k], []) for k in texts.keys()}
    total_length = len(concatenated_examples[list(texts.keys())[0]])
    total_length = (total_length // 512) * 512

    result = {k: [t[i: i + 512] for i in range(0, total_length, 512)]
              for k, t in concatenated_examples.items()}

    return result


def ptb_dataloader(tokenizer):
    all_data = load_dataset('ptb_text_only')
    all_data = all_data.map(ptb_encode,
                            remove_columns=['sentence'],
                            batched=True)
    all_data = all_data.map(ptb_group_texts,
                            batched=True)
    all_data.set_format(type="torch",
                        columns=['input_ids', 'attention_mask'])

    train_dataloader = DataLoader(all_data['train'],
                                  batch_size=8,
                                  shuffle=True,
                                  num_workers=1)

    val_dataloader = DataLoader(all_data['validation'],
                                batch_size=8,
                                num_workers=1)

    test_dataloader = DataLoader(all_data['test'],
                                 batch_size=8,
                                 num_workers=1)

    return train_dataloader, val_dataloader, test_dataloader