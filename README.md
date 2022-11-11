# Neural_Network_Charity_Analysis
Create a binary classifier using a deep neural network

## Overview

This analysis makes use of a deep neural network to analyze more than 34,000 organizations that have received funding from Alphabet Soup over the years. 

Using the features from the dataset, a binary classifier that is capable of predicting whether applicants will be successful if funded by Alphabet Soup will be created.

## Results

![Best_Model_Results](/images/Top_3_Models.png)

These are the results obtained after using keras tuner. The accuracy of the 3 top models fulfills the required 75% accuracy.

![Best_Model_Summary](/images/Top_Model_HP.png)

In this image, the structure of the best model is shown.

### Data Preprocessing

* **What variable(s) are considered the target(s) for your model?**

The target variable in this dataset is the *IS_SUCCESSFUL* feature. This feature represents the success (1) or failure (0) of the applicant.

* **What variable(s) are considered to be the features for your model?**

These features are considered to be relevant for the model:

    * NAME—Identification column
    * APPLICATION_TYPE—Alphabet Soup application type
    * AFFILIATION—Affiliated sector of industry
    * CLASSIFICATION—Government organization classification
    * USE_CASE—Use case for funding
    * ORGANIZATION—Organization type
    * STATUS—Active status
    * INCOME_AMT—Income classification
    * SPECIAL_CONSIDERATIONS—Special consideration for application
    * ASK_AMT—Funding amount requested

* **What variable(s) are neither targets nor features, and should be removed from the input data?**

The EIN column is dropped from the dataset, the name column already serves to identify the requester, thus making it redundant.

### Compiling, Training, and Evaluating the Model

* **How many neurons, layers, and activation functions did you select for your neural network model, and why?**

The optimal model has 2 hidden layers, the first one with 216 neurons and the second one with 43. The output layer has a single layer with a sigmoid activation function so that the output can be used as a binary classifier.

Since the dataset is not linearly separable, 2 layers were used as a first instance to measure the performance. In case the classifier did not meet the required accuracy, another hidden layer would've been used.

The number of neurons was determined using keras tuner. The first layer was given a value between 1 and the number of features, while the second layer was given a range from 1 to half the number of features. 

The activation functions were also chosen by keras tuner. The options were relu and tanh for the hidden layers, the output layer could only use the sigmoid activation function. 

* **Were you able to achieve the target model performance?**

The model was able to achieve the required performance, resulting in an accuracy of 78% after 20 epochs. This model was also trained for an additional 50 epochs however the accuracy decreased to 77.67%. Training it for 5 more epochs resulted in an accuracy of 77.88%. It's likely that the model started to overfit after the additional epochs.

* **What steps did you take to try and increase model performance?**

1. Drop only the EIN column to avoid having an entry that would be redudant

2. Bucket the NAME feature. 
    1. The value_counts for the NAME feature was saved, as well as the applicants name, were stored in a pandas DataFrame named name_freq.
    2. The name_freq DataFrame contains the times the applicant has received funding from AlphabetSoup. By making a value_counts of the counts in this DataFrame, what we end up with is the frequency of the requester's request. This count shows that out of the 19568 applicants, more than 17500 of these received funding once. While the remaining applicants have received funding previously. PARENT BOOSTER USA INC received funding 1260 times. 
    3. The upper bound using the IQR of the name_freq.COUNT.value_counts() is used as the threshold value with which the names are bucketed into Other, or left as is.

3. The Special Condierations feature is changed from an object dtype, into a numerical dtype. Reason being that his feature only had two values, Y and N, which can easily be changed into 1 and 0. Then it won't need enconding anymore and reduces the feature size by 1.

4. The ASK_AMT feature was analyzed to see if it should be bucketed, however the values were too spread out to be appropriately bucketed and it would result in a loss of information. The ASK_AMT minimum value was 5000, with a max of 8597806340. The IQR lower bound is 887.0 and the IQR upper bound is 11855.0. Moreover, 8206 requests were above the upper threshold, and 26093 were within bounds. Since the dataset is scaled, it won't be necessary to bucket this information.

5. The number of neurons was determined using keras tuner Hyperband to find the optimal value.

## Summary

The final model resulted in an accuracy of 78%, which is 3% above the requested accuracy. An alternative to this problem could be using a random forest classifier. The SVM might not be an optimal solution since the data is not easy to classify and it might not find an optimal hyperplane to classify these results.

Random Forest Classifiers work well with numerical and categorical features, which could reduce the required preprocessing.