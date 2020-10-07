# LT2316 H20 Assignment A1

Name: Xi Chen

Dear,
  I didn't know that you have left a comment here already last week, until yesterday I saw a little red dot in front of "Omdöme". I thought I would get a notification when you reply, so I dared not change any code before yesterday.
  After reading the instructions you added about feature extraction, I began to have more ideas about how everything should be. The added instructions were very useful.
  So, yesterday evening and today, I made a lot changes in the code, and I have run the main file, and everything worked fine. However, I'm not sure if I have done everything in the right direction or everyting just happen to run without errors.
  I understand that A1 is crucial for starting A2. So I think it's better you correct my assignment now, before I continue, in case everything I done was totally wrong.
  Thanks for helping!
/Xi Chen

## Notes on Part 1.

*fill in notes and documentation for part 1 as mentioned in the assignment description*

Found a special format in Train/DrugBank/Eszopiclone_ddi.xml, Charoffset has two values and ";" in the middle, i.e. two same words in the same sentence -> use find(token) to take the first one's position.

Some entity names consists of more than one word -> split to two words

Validation consists of 30% of Train:
val_ids = np.random.choice(train_ids, size = int(len(train_ids) * 0.3))


## Notes on Part 2.

*fill in notes and documentation for part 2 as mentioned in the assignment description*

Added argument: id2word=dataset.id2word

I choose the following features:

1. token_id, pre_token_id, next_token_id as word2vec

2. word_length can also be a important feature, since a lot drug names are in latin, and they can be long.  

3. last_letter of the word is the next feature I choose, I guess latin drug names may have a special feature that they ends with a/some certain letter. However, I'm not sure about that. ## It seems that I got an empty token in the database whose length is 0, therefore I got an error when I run len[-1]. I checked data_loading, but I couldn't figure out why. 

Others: POS can also be an important feature, however, I haven't figure out how to get the POS tags in a simpler way. Since we only have IDs in the database, we need to get the sentences first from the IDs. It seems that it will take a very long time to run the program in order to get the POS tags.

There can be other features, for example, how many "a"s, "e"s in the word, which can be tested in the future.

## Notes on Part Bonus.

*fill in notes and documentation for the bonus as mentioned in the assignment description, if you choose to do the bonus*

When I tried to plot a histogram, the results seemed very wrong. But it worked fine when I changed it to barplot.
