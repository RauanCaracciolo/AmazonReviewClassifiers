import pandas as pd
import scipy.sparse as sp
import re
from sklearn.feature_extraction.text import TfidfVectorizer
#Read the dataset, I didn't add the raw archive because he is too big, so the link is in DataLink.txt.
reviews_df = pd.read_csv('../data/Reviews.csv')
#Remove all the others columns, except Score and Summary.
reviews_df = reviews_df[['Score', 'Summary']].dropna()
#Remove the 3 scores, because this means 'neutral', and the binary classifier will just say positive and negative.
reviews_df = reviews_df[reviews_df['Score'] != 3]
#Transforms the values of score in positive(>3) and negative(<3).
reviews_df['Label'] = reviews_df['Score'].apply(lambda x: 1 if x > 3 else 0)
#Now, a function to remove punctuation and make the text lower case.
def clear_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]","", text)
    return text
#Aplly the function to all rows.
reviews_df['Summary'] = reviews_df['Summary'].apply(clear_text)
#Vectorize the data in numeric data.
vec = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
x = vec.fit_transform(reviews_df['Summary'])
#Get the label por the y.
y = reviews_df['Label']
#Save the data to use in another file.
sp.save_npz('../data/Reviews_prepared_x', x)
pd.DataFrame(y).to_csv("../data/Reviews_prepared_y.csv", index=False)