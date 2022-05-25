# birdclef-2022-3rd-place-solution

First of all, thanks to Kaggle and Cornell Lab of Ornithology for organizing this competition.

I was lucky enough to team-up with [UEMU](https://www.kaggle.com/asaliquid1011), it was a fruitful team-up with lots of ideas coming from the both sides. 
We've worked hard until the end and thanks to that we managed to secure 3rd place, congratz to all of the winners!

Now for our approach, the key points are the following:

* Use CNN proposed by [2nd place BirdCLEF 2021](https://www.kaggle.com/competitions/birdclef-2021/discussion/243463)
* Use SED model & training scheme proposed in [tattaka's 4th place solution](https://www.kaggle.com/competitions/birdclef-2021/discussion/243293)
* Use pseudo-labels & hand-labels in SED model training
* Divide scored birds into two sets and use different loss functions for training each one

Now let's get down to our best finding during competition, the bird split:

When we merged together, we looked on OOFs prediction produced by our models and found out, that SED w/ focal-loss performs very differently compared to mentioned CNN trained w/ BCELoss
