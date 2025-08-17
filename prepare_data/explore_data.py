#Huong Thu Le - Emma's work

# explore_data.py
import os
import re

#load data source in all categories
DATA_DIR = '../sorted_data_acl'
DOMAINS = ['books', 'dvd', 'electronics', 'kitchen_&_housewares']
LABEL_FILES = {
    'positive': 'positive.review',
    'negative': 'negative.review',
    'unlabeled': 'unlabeled.review',  # may not exist in some product category
}

REVIEW_PATTERN = re.compile(r'<review>(.*?)</review>', re.DOTALL)


def count_reviews_in_file(file_path: str) -> int:
    if not os.path.exists(file_path):
        return 0
    with open(file_path, encoding='utf-8') as f:
        content = f.read()
    return len(REVIEW_PATTERN.findall(content))


def read_sample(file_path: str, n_chars: int = 200) -> str:
    if not os.path.exists(file_path):
        return "(file not found)"
    with open(file_path, encoding='utf-8') as f:
        return f.read()[:n_chars]


def explore_dataset():
    print(f"Dataset root: {DATA_DIR}\n")

    grand_pos = grand_neg = grand_unl = 0

    for domain in DOMAINS:
        print(f"=== Domain: {domain} ===")
        domain_path = os.path.join(DATA_DIR, domain)

        pos_path = os.path.join(domain_path, LABEL_FILES['positive'])
        neg_path = os.path.join(domain_path, LABEL_FILES['negative'])
        unl_path = os.path.join(domain_path, LABEL_FILES['unlabeled'])

        pos_count = count_reviews_in_file(pos_path)
        neg_count = count_reviews_in_file(neg_path)
        unl_count = count_reviews_in_file(unl_path)

        grand_pos += pos_count
        grand_neg += neg_count
        grand_unl += unl_count

        print(f"Positive reviews: {pos_count}")
        print(f"Negative reviews: {neg_count}")
        if unl_count:
            print(f"Unlabeled reviews: {unl_count}")

        # Print review sample (maximum 200 characters)
        print("\nSample positive review:")
        print(read_sample(pos_path))

        print("\nSample negative review:")
        print(read_sample(neg_path))

        if unl_count:
            print("\nSample unlabeled review:")
            print(read_sample(unl_path))

        print("\n")

    print("=== Dataset summary ===")
    print(f"Total Positive: {grand_pos}")
    print(f"Total Negative: {grand_neg}")
    if grand_unl:
        print(f"Total Unlabeled: {grand_unl}")


if __name__ == "__main__":
    explore_dataset()
