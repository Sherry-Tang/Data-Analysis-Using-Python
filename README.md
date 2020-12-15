# Data-Analysis-Using-Python
Hi! Welcome to my github :wink:, This repository stores python project that I did.
## 1. Microsoft Malware Detection Prediction
datasource: kaggle\
model used: xgboost<br/> 

## 2. Text Summarization - T5 Transformer VS TextRank 

### Traditional Approach with NLTK
Teaditianal Approach generate summarization by generating similarity matrix and ranking all sentences by texkrank algorithm, and pick top sentences
- Step 1 -  Read text anc split it
- Step 2 - Generate Similary Martix across sentences
- Step 3 - Rank sentences in similarity martix
- Step 4 - Sort the rank and pick top sentences
- Step 5 - Output the summarize text
### Deep Learning Approach with T5
Deep learning model generate summary based on the semantic understanding of original documents, T5 was trained on C4 dataset-a comprehensive and large dataset developed by google, and it has both encoder and decoder blocks which guarantees its outstanding performance.
- Step 1 - load the local txt file, you can also integrate different python packages such as “Fransc” to transform other source of docuent into text file.
- Step 2 - Initiate the model, we used t5-large here. There are more advanced T5 versions such as t5-3b and t5-11b which will give you better performance, but they will also take a lot of RAM, so make sure that you have enough space, and it will also take a longer time.
- Step 3 -Load tokenizer tokenizer = T5Tokenizer.from_pretrained('t5-large')
- Step 4 - define the instruction by adding the prefix"Summarize", You can replace "Summarize" by "translate English to other languages:" to translate the summary into different language
- Step 5 - Encode text
- Step 6 - Generate summary
- Step 7 - Decoder the summary and print it out
### T5 is a good tool to improve your daily work efficiency! 😊

Welcome to reach out if you have any questions.
 
