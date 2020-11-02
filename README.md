# Corona_tweet_classifier
The project uses XLNet to classify corona related tweets. For this, XLNetForSequenceClassification model from Huggingface is used in Pytorch. The dataset is small but acheive a reasonably good model to classify tweets.

The size of the dataset is as below.

Train Dataset: 6,416

Eval Dataset: 2,137

After running the training module for 20 epoch with ADAM Optimizer (learning rate = 1e-4) the eval loss was found to be 0.20023754
