# NLP_StackOverFlowClassification
Implement various NLP Model To work with StackOverflowQuestion Dataset

for Thai lang version you can read it on my medium [article](https://medium.com/@phaiphon_m/%E0%B8%81%E0%B8%B2%E0%B8%A3%E0%B9%83%E0%B8%8A%E0%B9%89-nlp-with-stackoverflow-question-quality-classification-161e7e1abe44) 

![](https://media1.tenor.com/m/bkx7ADV8vm0AAAAd/request-to-chat-gpt-request.gif)


## Introduction
In this Project, we will apply machine learning knowledge in the area of ​​nlp (Natural language processing) to data to serve as a guide for building models. You can apply this knowledge to your own data in the future

## let have a look at our dataset
The dataset we have chosen today is [60k Stack Overflow Questions with Quality Rating](https://www.kaggle.com/datasets/imoore/60k-stack-overflow-questions-with-quality-rate). The information in the dataset This is a collection of questions from websites that we programmers know very well. stackoverflow has come to rate the quality of the various questions that people post.

### let have a closer look
![image](https://github.com/Supmanzz555/NLP_StackOverFlowClassification/assets/83536257/1c9ae545-b4af-4d09-a317-2121e26117c3)

we have 45000 sample (i cut it off so it not too much for my computer) and 6 feature with target named y which has 3 class

- HQ: High-quality posts without a single edit.
- LQ_EDIT: Low-quality posts with a negative score, and multiple community edits. However, they still remain open after those changes.
- LQ_CLOSE: Low-quality posts that were closed by the community without a single edit.

with no missing value nor any imbalanced problem

## Preprocess
we cut off ID, creationDate and Tags so the rest is title body and y

