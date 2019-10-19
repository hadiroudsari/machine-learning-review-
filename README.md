
# machine-learning-review(from mitchellmachine learning book and prof velardi power point slies)
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
 
