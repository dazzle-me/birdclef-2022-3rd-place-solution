# birdclef-2022-3rd-place-solution

First of all, thanks to Kaggle and Cornell Lab of Ornithology for organizing this competition.

I was lucky enough to team-up with [UEMU](https://www.kaggle.com/asaliquid1011), it was a fruitful team-up with lots of ideas coming from the both sides. 
We've worked hard until the end and thanks to that we managed to secure 3rd place, congratz to all of the winners!

Now for our approach, the key points to build strong and reliable pipeline are the following:

* Use SED model & training scheme proposed in [tattaka's 4th place solution](https://www.kaggle.com/competitions/birdclef-2021/discussion/243293)
* Use pseudo-labels & hand-labels in SED model training
* Use CNN proposed in [2nd place BirdCLEF 2021 solution](https://www.kaggle.com/competitions/birdclef-2021/discussion/243463)
* Divide scored birds into two sets and use different loss functions to train models for each one
* Augmentations

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

### CNN model training, Group1 birds (slime part)

First of all, I want to share some thought on CNN model. The Psi & team managed to create a pretty wonderful net, I was confused at first, - how is it different from simple fully-convolutional net which takes whole 30sec crop as an input? The difference is the following, - since before feeding 30 second crop into the backbone they reshaped it to 6x5sec segments, therefore they limited recieptive field of neural network to only 5-second crop, which I think helped generalize to 5 second crops during the inference time, even though the net itself was trained on 30sec crops, but it requires more experimentations to see the difference.
 
Since the CNN model was used only for inference on large enough classes, it allowed us to build reliable validation and monitor metrics for Group1 birds only, 
for these models we used BCE loss to select best models on validation, however with mix-up augmentation model converged on the last epoch, - this fact allowed us to include some models which were trained on full data in the final ensemble

#### Training strategy

* Epochs: 40
* Optimizer: Adam, lr=3e-4, wd=0
* Scheduler: CosineAnnealing w/o warm-up
* Labels: use union of primary and secondary labels
* Startify data: by primary label

#### Augmentations

The ones that definitely helped
* mix-up (the most impactful one)
* add background noise (same as 2nd place solution 2021)
* spec-augment
* cut-mix (helped, but just a little)

#### Didn't work

* augment only scored birds
* multiply loss for scored bird by 10 
* use weighted BCE w/ weights proportionally to number of class appearance in dataset
* random power as in [vlomme's 2021 solution](https://www.kaggle.com/competitions/birdclef-2021/discussion/243351)
* coord-conv as in [2nd place rainforest solution](https://www.kaggle.com/competitions/rfcx-species-audio-detection/discussion/220760)

### Final results

| Name                    | Public LB   | Private LB | 
| -----------             | ----------- | ---------- |
| CNN model (no augs)     | 0.7715      | 0.7278     |
| Best slime ensemble (CNN only w/ BCE loss)| 0.8327        | 0.7898 |
| Combine UEMU's SED w/ focal-loss & slime's CNN w/ BCE Loss using bird split mentioned above | 0.8532             | 0.8052 |
| Same as above, but add more CNNs w/ BCE loss and SED w/ BCE loss to group1 birds (best public LB, sub1) | 0.8750  | 0.8126 |
| Safe submission (sub2) | 0.8556 | 0.8071 |

We also had around 20 subs which score > 0.82 on private LB and > 0.87 on public LB, but we didn't select them since we chose two subs in the following manner, -
one is the best public LB and the other one with much lower thresholds to prevent the shake-up on group2 birds, this sub also happend to be placed 3rd, we're happy :)
