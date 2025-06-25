# Author Verification Repo

The repo began as a way to return the log probabilities of tokens from LLMs, but will now be used for not only that but for implementation and scoring methods in Python.

## Methods

Current methods implemented in the repo:

* LambdaG

## Pipeline

The pipeline is slightly different for paraphrased data compared to the datasets in the original LambdaG moehtod. The original data comes in the format of corpus folders with each containing a separate known and unknown author dataset. There is also a metadata dataset in the root of the folder which contains all problem info, this is the same for test and training. :

### Original Datasets

1. For both the known and unknown data we split the data into sentences. We do this slightly differently to the normal way of sentence spliiting. I observed names like 'Samuel L. Jackson' in the text of the corpus, which would be split over separate sentences. So we perform NER -> split sentnces -> replace the NER text.
2. If we are running methods like LambdaG we might use PoS tagging (not yet implemented).
3. We might also want to tokenize the data using an LLM.
4. For the original datasets, methods like LambdaG can take the known dataframe as references and it will sample from authors not in the known and unknown.

### Paraphrased Datasets

1. For the known documents we run each through an LLM asking it to paraphrase/generate n copies
2. we then score the new copies.
3. Keep the top k copies for this score.
4. Then we might perform the same sentence splitting observed above in Original Datasets-1.
5. We could then tokenize the sentences using the same model as used to paraphrase
6. Complete the methods.