
#basics
import pandas as pd
import torch

# Feel free to add any new code to this script


def extract_features(data:pd.DataFrame, max_sample_length:int, device):
    # this function should extract features for all samples and 
    # return a features for each split. The dimensions for each split
    # should be (NUMBER_SAMPLES, MAX_SAMPLE_LENGTH, FEATURE_DIM)
    # NOTE! Tensors returned should be on GPU
    #
    # NOTE! Feel free to add any additional arguments to this function. If so
    # document these well and make sure you dont forget to add them in run.ipynb
    train_sentences= []
    val_sentences= []
    test_sentences= []
    for sen_id in list(data["sentence_id"].unique()):
        sentence=[]
        s_tokens = data[data["sentence_id"]==sen_id]
        for i,t_row in s_tokens.iterrows():
            sentence.append(t_row["token_id"])
           
        split=s_tokens['split'].unique().tolist()[0]
        if split == "TRAIN":
            train_sentences.append(torch.Tensor(sentence))
        elif split == "VAL":
            val_sentences.append(torch.Tensor(sentence))
        elif split == "TEST":
            test_sentences.append(torch.Tensor(sentence))
    train_sentences = [torch.nn.functional.pad(i, pad=(0, max_sample_length - i.numel()), mode='constant', value=0) for i in train_sentences]
    val_sentences = [torch.nn.functional.pad(i, pad=(0, max_sample_length - i.numel()), mode='constant', value=0) for i in val_sentences]
    test_sentences = [torch.nn.functional.pad(i, pad=(0, max_sample_length - i.numel()), mode='constant', value=0) for i in test_sentences]
    #print(len(train_sentences),len(val_sentences),len(test_sentences))
    train_sentences_tensor=torch.stack(train_sentences).to(device=device)
    val_sentences_tensor=torch.stack(val_sentences).to(device=device)
    test_sentences_tensor=torch.stack(test_sentences).to(device=device)
    output_data=[train_sentences_tensor, val_sentences_tensor, test_sentences_tensor]
    return output_data
