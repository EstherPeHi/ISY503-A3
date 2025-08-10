# python_script/preprocessing.py
import matplotlib.pyplot as plt
import seaborn as sns

import os
import re
import pandas as pd
import unicodedata
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


class Preprocessor:
    def __init__(self, data_dir='sorted_data_acl', plot_dir='plot'):
        self.DATA_DIR = data_dir
        self.DOMAINS = ['books', 'dvd', 'electronics', 'kitchen_&_housewares']
        self.LABEL_MAP = {'positive.review': 1, 'negative.review': 0}
        self.plot_dir = plot_dir

        try:
            self.STOPWORDS = set(stopwords.words('english'))
        except LookupError:
            import nltk;
            nltk.download('stopwords')
            self.STOPWORDS = set(stopwords.words('english'))

        try:
            self.LEMMATIZER = WordNetLemmatizer()
        except LookupError:
            import nltk;
            nltk.download('wordnet')
            self.LEMMATIZER = WordNetLemmatizer()

    def load_reviews(self):
        """
        Real all review with label from raw files, return (reviews, labels)
        """
        reviews = []
        labels = []
        for domain in self.DOMAINS:
            for fname, label in self.LABEL_MAP.items():
                fpath = os.path.join(self.DATA_DIR, domain, fname)
                if not os.path.exists(fpath):
                    continue
                with open(fpath, encoding="utf-8") as f:
                    content = f.read()
                # Clean review and connect review with associated label <review>...</review>
                matches = re.findall(r'<review>(.*?)</review>', content, re.DOTALL)
                for review in matches:
                    reviews.append(review.strip())
                    labels.append(label)
        return reviews, labels

    def normalize_unicode(self, text):
        return unicodedata.normalize('NFKC', text)

    def remove_isbn(self, text):
        return re.sub(r'(\b\d{9,13}[xX]?\b)', '', text).strip(' ,;:')

    def remove_urls(self, text):
        return re.sub(r'http\S+|www\S+', '', text)

    def remove_emails(self, text):
        return re.sub(r'\S+@\S+', '', text)

    def remove_names(self, text):
        return re.sub(r'\b([A-Z][a-z]+\s?){1,3}\b', '', text)

    def clean_text(self, text):
        text = self.normalize_unicode(text)
        text = text.lower()
        text = self.remove_isbn(text)
        text = self.remove_urls(text)
        text = self.remove_emails(text)
        text = self.remove_names(text)
        text = re.sub(r'<.*?>', ' ', text)
        text = re.sub(r'[^a-z\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        words = [w for w in text.split() if 2 <= len(w) <= 20 and w not in self.STOPWORDS]
        words = [self.LEMMATIZER.lemmatize(w) for w in words]
        return ' '.join(words)

    def remove_duplicates(self, df):
        before = len(df)
        df = df.drop_duplicates(subset=['text'])
        after = len(df)
        print(f"Removed {before - after} duplicate reviews")
        return df

    def remove_outliers(self, reviews, labels, min_length=20, max_length=500):
        """
        Remove all reviews that are too short or too long based on the number of characters.
        Return: (filtered_reviews, filtered_labels)
        """
        filtered_reviews = []
        filtered_labels = []
        removed_count = 0

        for review, label in zip(reviews, labels):
            word_count = len(str(review).split())
            if min_length <= word_count <= max_length:
                filtered_reviews.append(review)
                filtered_labels.append(label)
            else:
                removed_count += 1

        print(f"Removed {removed_count} outlier reviews")
        return filtered_reviews, filtered_labels

    def check_data(self, reviews, labels):
        df = pd.DataFrame({'review': reviews, 'label': labels})

        print("\nClass distribution:")
        print(df['label'].value_counts(dropna=False))

        df['word_count'] = df['review'].apply(lambda x: len(x.split()))
        print("\nReview length statistics:")
        print(df['word_count'].describe())

        duplicates = df.duplicated(subset=['review']).sum()
        print(f"\nDuplicate reviews: {duplicates}")

        os.makedirs(self.plot_dir, exist_ok=True)
        plot_path = os.path.join(self.plot_dir, 'review_length_distribution.png')
        self.plot_review_length_distribution(df['review'].tolist(), save_path=plot_path)
        return df

    def plot_review_length_distribution(self, reviews, save_path=None):
        word_counts = [len(r.split()) for r in reviews]
        plt.figure(figsize=(12, 5))

        # Histogram
        plt.subplot(1, 2, 1)
        sns.histplot(word_counts, bins=50, kde=True, color='skyblue')
        plt.title('Histogram of Review Lengths (words)')
        plt.xlabel('Number of words')
        plt.ylabel('Count')

        # Boxplot
        plt.subplot(1, 2, 2)
        sns.boxplot(x=word_counts, color='lightgreen')
        plt.title('Boxplot of Review Lengths (words)')
        plt.xlabel('Number of words')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            print(f"Saved review length distribution plot to {save_path}")
        else:
            plt.show()
