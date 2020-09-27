
#basics
import random
import pandas as pd
import torch
import glob
import xml.etree.ElementTree as et
import numpy as np

import os

class DataLoaderBase:

    #### DO NOT CHANGE ANYTHING IN THIS CLASS ### !!!!

    def __init__(self, data_dir:str, device=None):
        self._parse_data(data_dir)
        assert list(self.data_df.columns) == [
                                                "sentence_id",
                                                "token_id",
                                                "char_start_id",
                                                "char_end_id",
                                                "split"
                                                ]

        assert list(self.ner_df.columns) == [
                                                "sentence_id",
                                                "ner_id",
                                                "char_start_id",
                                                "char_end_id",
                                                ]
        self.device = device
        

    def get_random_sample(self):
        # DO NOT TOUCH THIS
        # simply picks a random sample from the dataset, labels and formats it.
        # Meant to be used as a naive check to see if the data looks ok
        sentence_id = random.choice(list(self.data_df["sentence_id"].unique()))
        sample_ners = self.ner_df[self.ner_df["sentence_id"]==sentence_id]
        sample_tokens = self.data_df[self.data_df["sentence_id"]==sentence_id]

        decode_word = lambda x: self.id2word[x]
        sample_tokens["token"] = sample_tokens.loc[:,"token_id"].apply(decode_word)

        sample = ""
        for i,t_row in sample_tokens.iterrows():

            is_ner = False
            for i, l_row in sample_ners.iterrows():
                 if t_row["char_start_id"] >= l_row["char_start_id"] and t_row["char_start_id"] <= l_row["char_end_id"]:
                    sample += f'{self.id2ner[l_row["ner_id"]].upper()}:{t_row["token"]} '
                    is_ner = True
            
            if not is_ner:
                sample += t_row["token"] + " "

        return sample.rstrip()



class DataLoader(DataLoaderBase):


    def __init__(self, data_dir:str, device=None):
        super().__init__(data_dir=data_dir, device=device)


    def _parse_data(self,data_dir):
        # Should parse data in the data_dir, create two dataframes with the format specified in
        # __init__(), and set all the variables so that run.ipynb run as it is.
        #
        # NOTE! I strongly suggest that you create multiple functions for taking care
        # of the parsing needed here. Avoid create a huge block of code here and try instead to 
        # identify the seperate functions needed.
        data_df_cols = ["sentence_id", "token_id", "char_start_id", "char_end_id", "split"]
        data_df_rows = []

        ner_df_cols = ["sentence_id", "ner_id", "char_start_id", "char_end_id"]
        ner_df_rows = []

        vocab={}
        ner_dict={'O': 0, 'drug': 1, 'group': 2, 'brand': 3, 'drug_n': 4}

        #parse xml
        path=[['TRAIN','*/*/*.xml'],['TEST','*/Test for DrugNER task/*/*.xml']]
        for path in path:
            for file in glob.iglob(os.path.join(data_dir, path[1])):
                with open(file) as f:
                    xtree = et.parse(f)
                    xroot = xtree.getroot()

                for senten in xroot:
                    sentence_id = senten.attrib.get("id")
                    sentence = senten.attrib.get("text")
                    for token in sentence.lower().rstrip().split():
                        if token not in vocab:
                            vocab[token]=len(vocab)
                        token_id = vocab[token]
                        char_start_id = sentence.lower().find(token)
                        data_df_rows.append({"sentence_id": sentence_id, "token_id": token_id, 
                                    "char_start_id": char_start_id, "char_end_id": char_start_id+len(token),
                                    "split": path[0]})

                    for node in senten:
                        if node.tag == 'entity':
                            type_name = node.attrib.get("type")
                            if type_name in ner_dict: ner_id = ner_dict[type_name]
                            else: ner_id = 0
                            entity_name = node.attrib.get("text")
                            if len(entity_name.split(" ")) == 1:
                                ner_id = ner_dict[node.attrib.get("type")]
                                if ';' in node.attrib.get("charOffset"): # Deal with special format in Train/DrugBank/Eszopiclone_ddi.xml
                                    charOffset = [entity_name.find(token), entity_name.find(token) + len(token)]
                                else:
                                    charOffset = node.attrib.get("charOffset").split("-")
                                ner_df_rows.append({"sentence_id": sentence_id, "ner_id": ner_id, 
                                        "char_start_id": int(charOffset[0]), "char_end_id": int(charOffset[1]),
                                        })
                            else: #entity name consists of more than one word
                                for token in entity_name.split(" "):
                                    ner_id = ner_dict[node.attrib.get("type")]
                                    char_start_id = entity_name.find(token)
                                    ner_df_rows.append({"sentence_id": sentence_id, "ner_id": ner_id, 
                                        "char_start_id": char_start_id, "char_end_id": char_start_id + len(token),
                                        })

        #create dataframe
        self.data_df = pd.DataFrame(data_df_rows, columns = data_df_cols)
        self.ner_df = pd.DataFrame(ner_df_rows, columns = ner_df_cols)
        train_index = self.data_df[self.data_df["split"] == "TRAIN"].index
        
        #divide VAL
        val_index = np.random.choice(train_index, size = int(len(train_index) * 0.3))
        for i in val_index:
            self.data_df.at[i,"split"] = "VAL"
        
        self.id2word={value : key for (key, value) in vocab.items()}
        self.id2ner={value : key for (key, value) in ner_dict.items()}
        self.vocab=[[key,value] for key, value in vocab.items()]
        
        
        #create X and Y data
        self.train_sentences= []
        self.train_labels= []
        self.val_sentences= []
        self.val_labels= []
        self.test_sentences= []
        self.test_labels= []
        for sen_id in list(self.data_df["sentence_id"].unique()):
            s_tokens = self.data_df[self.data_df["sentence_id"]==sen_id]
            s_ners = self.ner_df[self.ner_df["sentence_id"]==sen_id]
            #s = [item["token_id"] for item in s_tokens]
            sentence=[]
            label = []
            for i,t_row in s_tokens.iterrows():
                sentence.append(t_row["token_id"])
                is_ner = False
                for i, l_row in s_ners.iterrows():
                     if t_row["char_start_id"] >= l_row["char_start_id"] and t_row["char_start_id"] <= l_row["char_end_id"]:
                        label.append(l_row["ner_id"])
                        is_ner = True
                if not is_ner:
                    label.append(0)

            for index, row in s_tokens.iterrows():
                if row["split"] == "TRAIN":
                    self.train_labels.append(label)
                    self.train_sentences.append(sentence)
                elif row["split"] == "VAL":
                    self.val_labels.append(label)
                    self.val_sentences.append(sentence)
                elif row["split"] == "TEST":
                    self.test_labels.append(label)
                    self.test_sentences.append(sentence)
        a=max([len(i) for i in self.train_sentences])
        b=max([len(i) for i in self.val_sentences])
        c=max([len(i) for i in self.test_sentences])
        self.max_sample_length = max([a,b,c])
        pass


    def get_y(self):
        # Should return a tensor containing the ner labels for all samples in each split.
        # the tensors should have the following following dimensions:
        # (NUMBER_SAMPLES, MAX_SAMPLE_LENGTH)
        # NOTE! the labels for each split should be on the GPU
        train_labels = -1*np.ones((len(self.train_labels), self.max_sample_length))
        for j in range(len(self.train_labels)):
            cur_len = len(self.train_labels[j])
            train_labels[j][:cur_len] = self.train_labels[j]
        val_labels = -1*np.ones((len(self.val_labels), self.max_sample_length))
        for j in range(len(self.val_labels)):
            cur_len = len(self.val_labels[j])
            val_labels[j][:cur_len] = self.val_labels[j]
        test_labels = -1*np.ones((len(self.test_labels), self.max_sample_length))
        for j in range(len(self.test_labels)):
            cur_len = len(self.test_lables[j])
            test_labels[j][:cur_len] = self.test_labels[j]
        output = torch.LongTensor([train_labels,val_labels,test_labels]).to(device)
        return output


    def plot_split_ner_distribution(self):
        # should plot a histogram displaying ner label counts for each split
        
        pass


    def plot_sample_length_distribution(self):
        # FOR BONUS PART!!
        # Should plot a histogram displaying the distribution of sample lengths in number tokens
        pass


    def plot_ner_per_sample_distribution(self):        
        # FOR BONUS PART!!
        # Should plot a histogram displaying the distribution of number of NERs in sentences
        # e.g. how many sentences has 1 ner, 2 ner and so on
        pass


    def plot_ner_cooccurence_venndiagram(self):
        # FOR BONUS PART!!
        # Should plot a ven-diagram displaying how the ner labels co-occur
        pass



