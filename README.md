# NLP_StackOverFlowClassification
Implement various NLP Model To work with StackOverflowQuestion Dataset

for Thai lang version you can read it on my medium [article](https://medium.com/@phaiphon_m/%E0%B8%81%E0%B8%B2%E0%B8%A3%E0%B9%83%E0%B8%8A%E0%B9%89-nlp-with-stackoverflow-question-quality-classification-161e7e1abe44) 

![](https://media1.tenor.com/m/bkx7ADV8vm0AAAAd/request-to-chat-gpt-request.gif)


## Introduction
In this Project, we will apply machine learning knowledge in the area of ​​nlp (Natural language processing) to data to serve as a guide for building models. You can apply this knowledge to your own data in the future
> 

## let have a look at our dataset
The dataset we have chosen today is [60k Stack Overflow Questions with Quality Rating](https://www.kaggle.com/datasets/imoore/60k-stack-overflow-questions-with-quality-rate). The information in the dataset This is a collection of questions from websites that we programmers know very well. stackoverflow has come to rate the quality of the various questions that people post.

### let have a closer look
![image](https://github.com/Supmanzz555/NLP_StackOverFlowClassification/assets/83536257/1c9ae545-b4af-4d09-a317-2121e26117c3)

we have 45000 sample (i cut it off so it not too much for my computer) and 6 feature with target named y which has 3 class

- HQ: High-quality posts without a single edit.
- LQ_EDIT: Low-quality posts with a negative score, and multiple community edits. However, they still remain open after those changes.
- LQ_CLOSE: Low-quality posts that were closed by the community without a single edit.

**with no missing value nor any imbalanced problem**


## Preprocess
we cut off ID, creationDate and Tags so the rest is title body and y

encoded target class to number  
```python
data['Y'] = data['Y'].map({'LQ_CLOSE':0,'LQ_EDIT':1,'HQ':2})
```

and merge title and text into one big feature so we can train it all 
```python
data['text'] = data['Title'] + ' ' + data['Body']
data = data.drop(['Title','Body'],axis=1)
```
then we use regularexpression to remove special character and make every alphabets to lowercase by create a function 

```python
import re # import regularexpression
def clean(text): 
    text = text.lower() 
    text = re.sub(r'[^(a-zA-Z)\s]','',text)
    return text
```
here is final result after all of preprocessing  which will look more cleaner now

![image](https://github.com/Supmanzz555/NLP_StackOverFlowClassification/assets/83536257/36c07468-93a8-40cb-8fa5-15f2364619d1)

next step is tokenize every word 

```python
from keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer(num_words=500) 
tokenizer.fit_on_texts(X)
vocab = tokenizer.word_index
```

then padding the length of all sequence

```python
from keras.preprocessing.sequence import pad_sequences
X_padded = pad_sequences(X_tokenized, maxlen=100, padding='post', truncating='post')
```

> numword and maxlen in both tokenize and padding function can change by how you test the data or simply visualize it which can impact performance of the model (i already tested it these value 500 and 100 are the best)


## create the model
i will show only 1 model which is LSTM (the one that give me best result) 

we spilt data 70/30 then when we create LSTM model we will use
- word embedding layer (to help capture relation in word sequence)
- dropout layer (act as regularization)
- and 1-2 hidden layer (our data is not that big so we try to keep model small sized)


here the code of how we will create LSTM model

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

model = Sequential()
model.add(Embedding(input_dim=500, output_dim=30, input_length=100)) 
model.add(Dropout(0.3)) #dropout layers
model.add(LSTM(units=5))
model.add(Dropout(0.3)) #dropout layers
model.add(Dense(units=3, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
```
for brief summary 
- input layer has dimension 500 and length = 100 (same as tokenized and padding)
- dropout has set to 30% droprate so that model that capture some pattern too much
- LSTM layer we will not set any higher number here because it can overfit anytime (but you can test)
- output layer with softmax activation which seems the best with our dataset (you can try another that compatible with muticlass classification)

let traning them!

```python
simple = model.fit(
    X_train, y_train,
    epochs=50, 
    batch_size=10, 
    validation_data=(X_test, y_test),
    callbacks=[early_stopping]
)
```
we will use early stoping to stop when model seem to overfitting 


and here the result

![image](https://github.com/Supmanzz555/NLP_StackOverFlowClassification/assets/83536257/50026a00-5f59-4a91-9c1a-f9faf8b303e4)

- early stopping never triggered (but i set it 15 if we set it 10 maybe it will stop)
- model donest appare to be any sign of overfit at all by inspect validation and trainning loss
- accuracy seem okay for both

here confusion matrix

![image](https://github.com/Supmanzz555/NLP_StackOverFlowClassification/assets/83536257/c6152dcf-e623-4317-9834-90a2a23c9d46)

- mostly are True case
- false case are around 13.27% or 1791 of all testcase (13500) which is good

## Ref
[DATASET](https://www.kaggle.com/datasets/imoore/60k-stack-overflow-questions-with-quality-rate)


