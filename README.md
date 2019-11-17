
# machine-learning-review(from mitchellmachine learning book and prof velardi power point slies
https://towardsdatascience.com/machine-learning-text-processing-1d5a2d638958 )
some elementary notes about machine learning

## Machine learning Design cycle:
 1. Data  
 2. Feature selection   
 3. Model Selection  
 4. learning  
 5. Ecaluation  
 
Model=algorithm(implemented with a program)

# Types Of Learning  algorithm

-----
## Supervised Learning Vs Unsupervised Learning vs Reinforcment learning 

### Supervised learning:
The kind of algorithm that has some kind of data set (index and attribute and has the lable for those attribute). for any entry data this algorithm check the attribute of new unlable data with the attribute of labled data, and if these two are match together it will lable the new data. 

### Algorithms:  

**Discrete Classifiers**
  * Decision Trees
  * Decision Forests
  * SVM   
  
 **Continous (Regression ) classifiers:**
  * Neural networks
  
 **Probability estimators:**
  * Naive Bayes  
----

### UnSupervised learning: 
 Unsupervised model does not have any label sample so it check the input behaviour and shape and cluster input to diferent output. 

### Algorithm:  
 * Convolutional Neural Network(semi-supervised)
 * Rule Learning (Apriri,FPGrows)
 * Clustring()
----

>>>
### Reinforment learninh
  a kinf of learning that system learn from different things based on penalties and rewards.
  **Algoritm for reinforcment learning**
   * Q-learning
   * Genetic algorithms
 ----   
  ------
 
 
 
  `Learning: optimization problem The idea is to learn a function that minimizes an error or one that maximizes reward over    penalty.`  
   Given a selected model, during the learning phase, several model ***hyper-parameters*** need to be optimized.  
   
**Hyper-parameter** is a parameter whose value must be set before the learning process begins. For example, the maximum depth of a decision tree, the number of hidden layers in a neural network, the type of kernel function in SVM, the type of similarity function in clustering, etc.
  
**Hyper-parameters** can significantly impact on performance: 
 `Suitable hyper-parameters must be determined for each task 
 Occur in both supervised and unsupervised learning` 
   
 ` need for disciplined and automated optimization methods` 

---

### Evaluation
1) simple method: Split the dataset into the training-set and test-set (e.g., 70%, 30%) 
2) Other more complex methods: roblem: We cannot be 100 % sure about an evaluation estimate on a sub-sample of data:
   Solution:Test the statistical significance of the results

----
----
----
----

Decision Trees is a popular discrete classifiers 

every ***node*** is a test on the value of one feature. For each possible outcome of the test an arrow is created that links to subsequent test node or to a leaf node.
leaf nodes are ***decision***  concerning to the value of classification. 

 **Top-Down Decision Tree Induction**  recursicly build a tree top-down by divide and conquer.  
 
 At each step,  we aim to find the ***best split*** of our data. What is a good split? One which reduces the uncertainty of   classification for “some” split!
 
 > The process ends when we can output decisions (= the class labels),
                       
----
  
  **Which Attribute Is the Best Classifier?** statistical property, called ***informution gain***, that
measures how well a given attribute separates the training examples according to
    their target classification. 
    
----       
The formula below represent the ***entropy***:  
<img class="ho tq gn n o gm ag gk" width="391" height="91" role="presentation" src="https://miro.medium.com/max/391/1*nNY_7_aWRwp8E2DyGduEPg.png">

----
  Information Gain:
  
  <img class="ho qc gn n o gm ag gk" width="507" height="60" role="presentation" src="https://miro.medium.com/max/1307/0*08CaHVjPCgs_fZyp">


 example of information gain page 35 dtree:
  
>  information gain from outlook = (intial entropy)  -   (  (weight of sunny) * (sunny entropy) + (weight of overcast) * (overcast entropy) + (weight of Rainy) * (Rainy entropy)  )   
  
------------------------------------------------------------
  
## Feature Engineering
### Feature extraction 
#### Step 1: Data Preprocessing
   * Tokenization
   * Removing unnecessary punctuation, tags
   * Removing stop words — frequent words such as ”the”, ”is”, etc. that do not have specific semantic
   * Stemming — words are reduced to a root by removing inflection through dropping unnecessary characters, usually a       suffix
   * Stemming — words are reduced to a root by removing inflection through dropping unnecessary characters, usually a suffix.
   * Lemmatization — Another approach to remove inflection by determining the part of speech and utilizing detailed database of the language.
```python
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
tokens = word_tokenize("The quick brown fox jumps over the lazy dog")
print(tokens)
```
---

     
#### step 2: Feature Extraction     
* Bag of Words(Term Frequency-Inverse Document Frequency (TF-IDF) technique.)
* Word Embedding(Word2Vec,Glove)   

#### Step 3: Choosing ML Algorithms
                          


-----------------------------------------------------------

#### Normalization: Scaling & Centering  
  Since the range of values of raw data varies widely, in some machine learning algorithms, objective functions will not work properly without normalization.
  both scaling and centering have  formula.

 
----

## Evaluation   

> Basically if we want to focus more on minimising False Negatives, we would want our Recall to be as close to 100% as possible without precision being too bad and if we want to focus on minimising False positives, then our focus should be to make Precision as close to 100% as possible([link](https://medium.com/thalus-ai/performance-metrics-for-classification-problems-in-machine-learning-part-i-b085d432082b)).


<img class="ho qc gn n o gm ag gk" width="507" height="507" role="presentation" src="https://miro.medium.com/max/931/1*5XuZ_86Rfce3qyLt7XMlhw.png">  
  
  <img class="ho qc gn n o gm ag gk" width="507" height="507" role="presentation" src="https://miro.medium.com/max/1250/1*KhlD7Js9leo0B0zfsIfAIA.png">  
    <img class="ho qc gn n o gm ag gk" width="507" height="507" role="presentation" src="https://miro.medium.com/max/1349/1*a8hkMGVHg3fl4kDmSIDY_A.png">   
    <img class="ho qc gn n o gm ag gk" width="507" height="507" role="presentation" src="https://miro.medium.com/max/742/1*deegiX75imQsVXYVpG_SDQ.png">     
    
F1 Score:
We don’t really want to carry both Precision and Recall in our pockets every time we make a model for solving a classification problem. So it’s best if we can get a single score that kind of represents both Precision(P) and Recall(R).
One way to do that is simply taking their arithmetic mean. i.e (P + R) / 2  
<img class="ho qc gn n o gm ag gk" width="507" height="507" role="presentation" src="https://miro.medium.com/max/550/1*W2CxvU7m8R6cB_oz2U3ouA.png">

F1 Score = 2 * Precision * Recall / (Precision + Recall)
