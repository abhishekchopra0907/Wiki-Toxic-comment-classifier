# Wiki-Toxic-comment-classifier
# What is it ?
This basically a model which classifies toxic wikipedia comments into 6 classes bases on how toxic they are
# Why ?
Being anonymous over the internet can sometimes make people say nasty things that they normally would not in real life. Let's filter out the hate from our platforms one comment at a time.
# How does it work ?
Basically this contains 2 NLP models-
  * a normal deep nueral network to classify our tokenised comments 
  * a LSTM  with one dense layer to classify our tokenised comments 
# Dataset
The dataset here is from wiki corpus dataset which was rated by human raters for toxicity
The types of toxicity are:
 * toxic
 * severely_toxic
 * obscene
 * threat
 * insult
 * identity_hate
#### The data set contained the following files:
* train.csv - the training set, contains comments with their binary labels
* test.csv - the test set, you must predict the toxicity probabilities for these comments. To deter hand labeling, the test set contains some comments which are not included in scoring.
# How to implement/run it ?
#### My code is basically the model architecture for the implementations of malaria detecting using CNN
* you can run this code by downloading the dataset provided in this repo and then training the model provided in the Wiki_Toxic_comments_classifier ipynb file on that dataset by just executing the Wiki_Toxic_comments_classifier ipynb file 
* make sure that the dataset and the Wiki_Toxic_comments_classifier ipynb file are saved in the same folder for proper implmentation

#### you can get the output/predicitions on unseen images by using the following code:
```python

comment_input=["ENTER COMMENT","ENTER COMMENTS","ADD INFO HERE","ENTER COMMENTS HERE"] ##list of comments you want to judge
comments_int=[]
for comment in comment_input:
    r = [vocab_to_int[w] for w in comment.split()]
    comments_int.append(r)
comments_len = [len(x) for x in comments_int]
comments_int = [ comments_int[i] for i, l in enumerate(comments_len) if l>0 ]
x=pad_features(comments_int,200)
re_x=reshaper(x)
model.predict(re_x)
```
in order to get the prediction on a set a comments , just initialize comments_input with those set of inputs and run the code mentioned above




