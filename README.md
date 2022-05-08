# forcedTextSimilarity


## existing text similarity flow
#### 1. load the dataset
#### 2. pre-process and create the tokens
#### 3. transform tokens to vector
#### 4. use the cosine similarity

These steps workes decent when we have good tokenizer and which builds up the Semantic relationships. But given situation where we need to make model learn beyound the semantitcs similarity we need to tweak some token weights and thats were the 'forcedTextSimilarity' comes to play.


After learning token weights from the selected tokenizer, this module will add the adjusting weights to each token which will force the cosine similarity to be approx. 1.0

NOTE : even we can force the cosine similarity to the required score i.e., 0.5 / 0.7 / 0.8 any number.
