# Amazon Review Classifiers
This project contains an ML solution for predicting if an product review is positive or negative.
## Tecnologies 🚀
* **Python:** Main proggraming language.
    ![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
* **Scikit-learn:** Machine learning Library, used to process the data, train and evaluate the models.
    ![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
* **Pandas:** Data manipulation Library.
    ![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
* **Matplotlib:** Data visualization Library.
    ![Matplotlib](https://img.shields.io/badge/Matplotlib-315A9E?style=for-the-badge&logo=matplotlib&logoColor=white)

## The process 💻

### Data preparation.
In this project, we use the public dataset found in (https://www.kaggle.com/datasets/arhamrumi/amazon-product-reviews).

First, we need to treat the data to utilize in a machine learning model.

1. The dataset has too much useless data, so we will remove all the colluns, except for the useful ones, in this case, "Rating", and chose one from "Summary" or "Review". I chose Summary, for less text and less data to process because my hardware is relatively weak.
2. Now, we have data in two types, rating from 1-5 and a summary, first, we want an binary classification in this case, so we will make reviews with rating > 3 positives and reviews with < 3 negatives, reviews with exact rating = 3 will be removed from the dataset, because they are "neutral", and useless in a binary classification.
3. After the rating, we need to transform the text in usefull data, so we use the Text Vectorization technique.
4. Finally, we have an ready dataset to be used to train an ML model, with features and a label.

### Evaluating and comparing models
In this project, i put a little more efort to create two classes to evaluate the binary classification models, one returns a text evaluation and the other a visual evaluation with pyplot, feel free to utilize this generic classes, in the next projects i made with this type of classification i will use them too.

<img width="1366" height="657" alt="Figure_1" src="https://github.com/user-attachments/assets/384774bc-7b04-4c04-978f-d918749afac8" />



