<!-- <h1 align="center">
<img src="https://gitlab.com/anna.giabelli/TaxoSS/-/blob/master/img/logo.svg" alt="TaxoSS" width="400">
</h1> -->
<h1 align="center">SEEDOT</h1>

<p align="center">
  <a href="#description">Description</a> •
  <a href="#installation">Installation</a> •
  <a href="#usage">Usage</a>
</p>

---

## Description

The **seedot** project aims to overcome the limitations of lexicon-based sentiment analysis tools, such as VADER, which are too general and not suitable for specific contexts. It addresses issues like word ambiguity, where a term may have different meanings based on the domain.
For example, "bull" can refer to an animal or indicate positive growth in the financial domain. 
The package consists of two main parts that lay the foundation for a comprehensive system capable of recognizing and analyzing specific discussion topics.

The **seedot** package provides specialized dictionaries for sentiment analysis in the following domains:
- Food: food review on amazon
- Electronics: review of electronic products on amazon
- Hotel: review of amusement parks on tripadvisor
- Finance: reviews and tweets of financial topics

It also provides the possibility to create your own domain oriented sentiment dictionaries.

The key takeaway is that using specialized dictionaries trained on specific lexicons consistently improves the performance of VADER for sentiment analysis. Later on more domain will be added.

We encourage you to explore the insights and input provided by this project, which involves developing a system capable of performing accurate analysis using specialized dictionaries. Furthermore, if you have specialized dictionaries that you would like to contribute, we welcome your collaboration to expand the range of options provided by **seedot**.

## Requirements

- Python 3.6 or later
- pandas
- os
- re
- math
- string
- codecs
- json
- itertools 
- inspect 
- io 
- nltk
- tqdm
- numpy as np
- sklearn
- json
- gensim

## Installation

**seedot** can be installed through `pip` (the Python package manager) in the following way:

```bash
pip install seedot
```

# Usage

## Functions

The **seedot** package is designed to offer the same functionalities as the VADER package while also enabling the ability to invoke specific dictionaries for sentiment analysis in different domains.

The **seedot** package is a powerful tool for performing sentiment analysis in specific domains. It offers two main functions that allow you to leverage domain-specific dictionaries for accurate sentiment analysis:

- `seedot_procedure()`: a function that allow you to build a domain specific dictionary for sentiment analysis starting form a dataset. 
- `sentiment_analyzer()`: a function to assigns a sentiment intensity score to sentences. This score quantifies the sentiment expressed in a sentence, indicating the level of positivity or negativity.

```python
from seedot import seedot_procedure
from seedot import sentiment_analyzer
```
### seedot_procedure function

The function `seedot_procedure()` is callable as it follow:

```python 
from seedot import seedot_procedure as sp
```

This function has two callable sub-functions:

- `score_functions()`: allow to get the score function division that allow to transform the model weight in score for vader dictionary. 
- `seedot()`: allow to apply the seedot procedure to get a domain oriented dictionary for sentiment analysis starting from a dataset.

#### score_functions()
The `score_functions()` allow to find the best cut to transform weights in scores following two different scoring functions.
It work with a sub-sub-function `division()` in the following way:

```python 
sp.score_functions(data = None, score_function = None, model = "logistic").division()
```

where:

- `data`: must be a dataset with two columns relatively called 'Score' (that contain: 'Positive', 'Negative', 'Neutral') and 'Text' that contain the text. It is `None` by default.
- `score_function`: must be ether `2` or `3`, referring the score function to use, it is `None` by default:

  -`2`: find the optimal division where to assign score 4/-4
  -`3`: find the optimal division where to assign score 4/-4 and 3/-3

- `model`: allow to specify the model to use for the procedure, currently only logistic has been implemented. It is `"logistic"` by default.

This function return a list that contain the accuracy value that identify the optimal cut, and the point where the points where to cut.
Lets see some examples:

```python 
d2 = sp.score_functions(data = df, score_function = 2).division()
d2

Maximum accuracy value ->  87.44 
Optimal division ->  +4: 0.4   -4: -0.4
(87.44, 0.4)
```

```python 
d3 = sp.score_functions(data = df, score_function = 3).division()
d3

Maximum accuracy value ->  89.04 
Optimal division ->  +4: 0.3   +3: 0.1   -4: -0.3   -3: -0.1
(89.04, 0.3, 0.1)
```
This function allow to understand some output for the `seedot()` function.

#### seedot()
The `seedot()` allow to create a dictionary with word and scores to be used for sentiment analysis.
It work with a sub-sub-function `domain_dictionary()` in the following way:

```python 
sp.seedot(data = None,
                       tokens = 2000, 
                       model = "logistic", 
                       score_function = 1, 
                       division = None, 
                       embedding = None).domain_dictionary()
```

where:

- `data`: must be a dataset with two columns relatively called 'Score' (that contain: 'Positive', 'Negative', 'Neutral') and 'Text' that contain the text. It is `None` by default.
- `tokens`: is the maximum number of feature (words) to extrapolate form the dataset, and which will update the vader dictionary.
- `model`: allow to specify the model to use for the procedure, currently only logistic has been implemented. It is `"logistic"` by default.
- `score_function`: must be ether `1`, `2` or `3`, referring the score function to use, it is `1` by default:

  -`1`: (-2: -4) (-1:-3.5) (-0.5, -2.5) (0.5: +2.5) (1: +3.5) (2: +4)
  -`2`: division to assign score 4/-4
  -`3`: division to assign score 4/-4 and 3/-3

- `division`: linked to `score_function` value, can be ether `None` or a list with 1 or 2 element identifying the cuts:

  - `score_function = 1`: must be `None`
  - `score_function = 2`: must be a one element list with the cut for 4 scores, for example `[0.6]`
  - `score_function = 3`: must be a two element list with the cut for 4 in position 0 and the cut for 3 in position 1, for example `[0.8, 0.2]`

- `embedding`: allow to specify a procedure to change the scores of similar words, that already have a score in vader, can be `None`, `1` or `2`, it is `None` by default:

  - `None`: no procedure is applied
  - `1`: the scores of the most similar words to the ones that have score +4 or -4 are set equal to +4 or -4
  - `2`: the score of the most similar words to the ones that have score +4 or -4 are changed with the mean between the score of the similar word (+4 or -4) and the score of the word selected in vader dictionary

This function return a dictionary with words and scores that can be used to perform sentiment analysis.
Lets see some examples:

```python 
dic = sp.seedot(data = df, tokens =10, score_function = 3, division = [0.3, 0.1], embedding = None).domain_dictionary()
count = 0
for key, value in dic.items():
    print(key, value)
    count += 1
    if count >= 10:
        break

seedot_dictionary has been created.
$: -1.5
%) -0.4
%-) -1.5
&-: -0.4
&: -0.7
( '}{' ) 1.6
(% -0.9
('-: 2.2
(': 2.3
((-: 2.1
```

```python 
em_dic = sp.seedot(data = df, tokens =10, score_function = 3, division = [0.3, 0.1], embedding = 2).domain_dictionary()

seedot_dictionary has been created.
```

This function allow to create a domain oriented dictionary to be used or the `sentiment_analyzer()` function.

### sentiment_analyzer function

The function `sentiment_analyzer()` is callable as it follow:

```python 
from seedot import sentiment_analyzer as sa
```

This function has a callable sub-functions:

- `SIA()`: SentimentIntensityAnalyzer()  function assigns a sentiment intensity score to sentences. This score quantifies
the sentiment expressed in a sentence, indicating the level of positivity or negativity.

#### SIA()
The `SIA()` allow to create a perform sentiment analysis selecting a domain specific dictionary.
It work with a many sub-function, sub-functions that have been kept the same as those found in VADER, ensuring compatibility and familiarity. For detailed information and exploration of all the possibilities, we recommend referring to the official VADER repository on GitHub: <https://github.com/cjhutto/vaderSentiment>.

Here in order to show `SIA()` function we will use the `polarity_scores()` subfunction that allow to ger the sentiment scores of a phrase.

```python 
sa.SIA(domain = None, emoji_lexicon="emoji_utf8_lexicon.txt").polarity_scores("Seedot is the best")
```

where:

- `domain`: can bee one of the following domain `['food', 'finance', 'electronic', 'hotel']` that lead to a dictionary for the domain specified or a dictionary obtained by a dataframe with the `seedot_procedure.seedot()` function, it is `None` by default.
- `emoji_lexicon`: specify the emoji lexicon to use, it is pre-specified by default 

This function allow to perform domain oriented sentiment analysis.
Lets see some examples:

```python 
sa.SIA(domain = "food").polarity_scores("Seedot is yummy")

{'neg': 0.0, 'neu': 0.375, 'pos': 0.725, 'compound': 0.7184}
```

```python 
sa.SIA(domain = "finance").polarity_scores("Seedot is breaking the market")

{'neg': 0.0, 'neu': 0.5, 'pos': 0.5, 'compound': 0.8184}
```

```python 
sa.SIA(domain = "electronic").polarity_scores("Seedot is shocking")

{'neg': 0.574, 'neu': 0.426, 'pos': 0.0, 'compound': -0.4019}
```

```python 
sa.SIA(domain = "hotel").polarity_scores("Seedot is clean")

{'neg': 0.0, 'neu': 0.333, 'pos': 0.667, 'compound': 0.6124}
```

```python 
dic = t.seedot(data = df, tokens =10, score_function = 1, division = None, embedding = 1).domain_dictionary()
sa.SIA(domain = dic).polarity_scores("Seedot is personal")

{'neg': 0.0, 'neu': 0.417, 'pos': 0.583, 'compound': 0.6369}
```

With the **seedot** package, you can effortlessly perform sentiment analysis in various domains, unlocking valuable insights from text data.

