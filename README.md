# birdclef-2022-3rd-place-solution

First of all, thanks to Kaggle and Cornell Lab of Ornithology for organizing this competition.

I was lucky enough to team-up with [UEMU](https://www.kaggle.com/asaliquid1011), it was a fruitful team-up with lots of ideas coming from the both sides. 
We've worked hard until the end and thanks to that we managed to secure 3rd place, congratz to all of the winners!

Now for our approach, the key points are the following:

* Use CNN proposed in [2nd place BirdCLEF 2021 solution](https://www.kaggle.com/competitions/birdclef-2021/discussion/243463)
* Use SED model & training scheme proposed in [tattaka's 4th place solution](https://www.kaggle.com/competitions/birdclef-2021/discussion/243293)
* Use pseudo-labels & hand-labels in SED model training
* Divide scored birds into two sets and use different loss functions to train models for each one

Now let's get down to our best finding during competition

### The bird split

When we merged together, we looked on OOFs prediction produced by our models and found out that SED w/ focal-loss performs very differently compared to mentioned CNN trained w/ BCELoss depending on number of training samples

From our observations, SED models w/ focal-loss tend to make more conservative predictions, and due to the loss design they don't miss small classes:

Here in blue you can see SED model w/ focal-loss result, and in pink CNN model trained w/ BCE loss
 
![image](https://user-images.githubusercontent.com/57013219/170329125-532a0640-cb54-4a81-9d8a-fadd4721d6ae.png)

The same can be said about advantage of models which used BCE loss during training over models with focal-loss for the large classes 
![image](https://user-images.githubusercontent.com/57013219/170329065-9a9d4da1-1660-46d4-b25e-9419f451f63d.png)

Therefore, we divided the birds into two groups according to the number of data and manual inspection of probability density functions of target-data as above, and used different models for them.

Group1: 14birds  ['jabwar', 'yefcan', 'skylar', 'akiapo', 'apapan', 'barpet', 'elepai', 'iiwi', 'houfin', 'omao', 'warwhe1', 'aniani', 'hawama', 'hawcre'], 

for them we ended up using CNN + SED which were trained using BCE loss.

Group2: 7birds,  ['crehon', 'ercfra', 'hawgoo', 'hawhaw', 'hawpet1', 'maupar', 'puaioh'], for this group we chose to use SED w/ focal-loss.



