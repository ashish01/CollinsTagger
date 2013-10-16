CollinsTagger
=============

This is an implementation of [Collins Perceptron](http://www.cs.columbia.edu/~mcollins/papers/tagperc.pdf) for structured prediction. It is an online learning algorithm which learns the parameters of the model by doing simple perceptron updates. This implementation borrows a couple of tricks from [Vowpal Wabbit](https://github.com/JohnLangford/vowpal_wabbit/wiki) to keep the model size and runtime memory use bounded irrespective of number of feature used.

Setup
=====
The project is completely self-contained and only depends on .NET 3.5 and complies with both Visual Studio and Mono. To start you can either clone this repository and build or you can [downlond](https://github.com/ashish01/CollinsTagger/releases/download/v0.1/CollinsTagger.exe) the binary from the [initial release](https://github.com/ashish01/CollinsTagger/releases/tag/v0.1).

The CollinsTagger.exe works on Windows with .NET 3.5 or higher installed and requires mono runtime on *nix like operating systems.

Example
=======
I will use [tutorial](http://www.chokkan.org/software/crfsuite/tutorial.html) from the excellent [CRFSuite](http://www.chokkan.org/software/crfsuite/) project here. The goal is to build a model which predicts chunk labels for a given sentence.

Training and testing data
-------------------------
This tutorial uses training and testing data distributed by [CoNLL 2000 shared task](http://www.cnts.ua.ac.be/conll2000/chunking/). Download the files using these commands
```
mkdir example
cd example
curl -O http://www.cnts.ua.ac.be/conll2000/chunking/train.txt.gz
curl -O http://www.cnts.ua.ac.be/conll2000/chunking/test.txt.gz
```

Featureization
--------------
The feature generation is exactly same as done in CRFSuite and uses the same scripts. So let’s get the featureization script from CRFSuite's repository
```
curl -O https://raw.github.com/chokkan/crfsuite/master/example/chunking.py
curl -O https://raw.github.com/chokkan/crfsuite/master/example/crfutils.py
```

Now featureize the train and test data
```
gunzip *
python ./chunking.py < train.txt > train.crf.txt
python ./chunking.py < test.txt > test.crf.txt
```

Now we are ready to train a chunking model on this data.

Model Training
--------------
While training a model you have a choice on how you want to convert feature names to feature ids. One way is to assign a unique integer to each feature string and store this mapping in a Dictionary. Advantage of this approach is that there is no feature collision and all your features are preserved. Disadvantage is that the size of dictionary will increase proportionally to the number of features. This could be a problem if the number of features are very large.

To keep runtime memory size bounded, irrespective of number of features, you can use the second approach - feature hashing. Because of hashing different features can hash to same feature id which may or may not cause some loss of performance. But empirically this should be minor.

Let’s start with a conventional model.
```
./CollinsTagger.exe train --basePath . --numIterations 10 --data train.crf.txt
```

If all goes well, the program should print average loss with exponential back off. Loss is defined as the number of tags it gets wrong. Let’s walk through the above command

```
CollinsTagger.exe - The main program

train - the mode of operation. There are only two modes - train and tag

basePath - the folder where all the model files are saved (in case of train mode) or read (in case of tag)

numIterations - number of times to iterate the data (only used during training)

data - input data in CRFSuite format
```

The run above will produce 4 model files
```
TagTransitionProbabilities.txt - weight of current tag given previous tag
TagDictionary.txt - mapping of tag names to tag ids
FeatureDictionary.txt - mapping of feature names to feature ids (not created when using feature hashing)
TagFeatureProbabilities.txt - weight of current tag given different features
```

Prediction
----------
Now let’s see how well this model learned by seeing its performance on test data.
```
CollinsTagger.exe tag --basePath . --data test.crf.txt --output output.txt
```

The command above is almost same as train command except the mode is different and we provide an additional argument called output. The output file will contain the tags as predicted by the model. It will also print the precision/recall statistics per tag and in the last line it prints the per instance accuracy.

Here is the snapshot of what I get in my local run

```
B-NP    12411   12038   12422   0.969946015631295       0.969087103526002
B-PP    4870    4702    4811    0.965503080082135       0.977343587611723
I-NP    14426   13962   14376   0.967835851934008       0.971202003338898
B-VP    4677    4475    4658    0.956809920889459       0.960712752254186
I-VP    2674    2541    2646    0.950261780104712       0.96031746031746
B-SBAR  513     451     535     0.879142300194932       0.842990654205607
O       6150    5952    6182    0.96780487804878        0.962795211905532
B-ADJP  413     330     438     0.799031476997579       0.753424657534247
B-ADVP  839     707     866     0.842669845053635       0.816397228637413
I-ADVP  80      51      89      0.6375                  0.573033707865169
I-ADJP  136     110     167     0.808823529411765       0.658682634730539
I-SBAR  19      3       4       0.157894736842105       0.75
I-PP    43      34      48      0.790697674418605       0.708333333333333
B-PRT   103     79      106     0.766990291262136       0.745283018867924
B-LST   0       0       5       NaN                     0
B-INTJ  2       1       2       0.5                     0.5
I-INTJ  0       0       0       NaN                     NaN
B-CONJP 8       5       9       0.625                   0.555555555555556
I-CONJP 13      10      13      0.769230769230769       0.769230769230769
I-PRT   0       0       0       NaN                     NaN
B-UCP   0       0       0       NaN                     NaN
I-UCP   0       0       0       NaN                     NaN
1160    2012    0.576540755467197
```

Feature Hashing
---------------
Now let’s train on same data using feature hashing. To train with feature hashing use this command
```
CollinsTagger.exe train --basePath . --numIterations 10 --data train.crf.txt --useFeatureHashing 18
```

useFeatureHashing uses feature hashing instead of feature to id map and 18 determines the number of bits used. More bits will create a bigger model and reduce feature collision.

add the same parameter while tagging
```
CollinsTagger.exe tag --basePath . --data test.crf.txt --output output.txt --useFeatureHashing 18
```

Here is the sample output
```
B-NP    12412   12027   12422   0.968981630679987       0.968201577845758
B-PP    4873    4715    4811    0.967576441617074       0.980045728538765
I-NP    14409   13950   14376   0.968144909431605       0.970367278797997
B-VP    4685    4475    4658    0.955176093916756       0.960712752254186
I-VP    2678    2542    2646    0.949215832710978       0.960695389266818
B-SBAR  508     453     535     0.891732283464567       0.846728971962617
O       6154    5950    6182    0.966850828729282       0.962471692009059
B-ADJP  404     323     438     0.79950495049505        0.737442922374429
B-ADVP  855     718     866     0.839766081871345       0.829099307159353
I-ADVP  85      55      89      0.647058823529412       0.617977528089888
I-ADJP  130     105     167     0.807692307692308       0.62874251497006
I-SBAR  17      3       4       0.176470588235294       0.75
I-PP    41      34      48      0.829268292682927       0.708333333333333
B-PRT   104     84      106     0.807692307692308       0.792452830188679
B-LST   0       0       5       NaN                     0
B-INTJ  1       1       2       1                       0.5
I-INTJ  0       0       0       NaN                     NaN
B-CONJP 8       5       9       0.625                   0.555555555555556
I-CONJP 13      10      13      0.769230769230769       0.769230769230769
I-PRT   0       0       0       NaN                     NaN
B-UCP   0       0       0       NaN                     NaN
I-UCP   0       0       0       NaN                     NaN
1156    2012    0.57455268389662
```

As you can see the loss of accuracy with feature hashing is not much. Try this you want to cap memory usage while having large number of features.

Enjoy!
