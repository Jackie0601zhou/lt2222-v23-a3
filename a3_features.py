import os
import sys
import argparse
import numpy as np
import pandas as pd
import re
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from typing import List, Tuple
# Whatever other imports you need

# Remove email headers and signatures from the given email string
def strip_headers_and_signature(email: str) -> str:
    email_parts = re.split(r'\n\n', email, maxsplit=1)
    if len(email_parts) < 2:
        return email
    email_body = email_parts[1]
    email_body_no_signature = re.split(r'--\s*\n', email_body)[0]
    return email_body_no_signature


def process_author_folders(data_path: str) -> List[Tuple[str, str]]:
    author_emails = []
    for author_folder in os.listdir(data_path):
        folder_path = os.path.join(data_path, author_folder)
        if os.path.isdir(folder_path):
            for email_file in os.listdir(folder_path):
                email_file_path = os.path.join(folder_path, email_file)
                with open(email_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    email_content = f.read()
                processed_email = strip_headers_and_signature(email_content)
                author_emails.append((author_folder, processed_email))
    return author_emails

def vectorize_and_reduce_emails(emails: List[Tuple[str, str]], n_components: int) -> List[Tuple[str, List[float]]]:
    authors, texts = zip(*emails)
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(texts)
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X.toarray())
    return list(zip(authors, X_reduced))


def split_train_test(data: List[Tuple[str, List[float]]], train_ratio: float = 0.8) -> Tuple[List[Tuple[str, List[float]]], List[Tuple[str, List[float]]]]:
    random.shuffle(data)
    split_index = int(len(data) * train_ratio)
    train_data = data[:split_index]
    test_data = data[split_index:]
    return train_data, test_data


def write_to_file(data: List[Tuple[str, List[float]]], output_file: str, data_type: str):
    with open(output_file, 'a') as f:
        for author, features in data:
            f.write(f"{data_type},{author},{','.join(map(str, features))}\n")

def main(args):
    emails = process_author_folders(args.inputdir)
    reduced_emails = vectorize_and_reduce_emails(emails, args.dims)
    train_data, test_data = split_train_test(reduced_emails, train_ratio)
    with open(args.outputfile, 'w') as f:
        f.write("type,author,features\n")

    write_to_file(train_data, args.outputfile, "train")
    write_to_file(test_data, args.outputfile, "test")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert directories into table.")
    parser.add_argument("inputdir", type=str, help="The root of the author directories.")
    parser.add_argument("outputfile", type=str, help="The name of the output file containing the table of instances.")
    parser.add_argument("dims", type=int, help="The output feature dimensions.")
    parser.add_argument("--test", "-T", dest="testsize", type=int, default="20", help="The percentage (integer) of instances to label as test.")

    args = parser.parse_args()

    print("Reading {}...".format(args.inputdir))
    emails = process_author_folders(args.inputdir)

    print("Constructing table with {} feature dimensions and {}% test instances...".format(args.dims, args.testsize))
    reduced_emails = vectorize_and_reduce_emails(emails, args.dims)
    train_ratio = 1.0 - args.testsize / 100
    train_data, test_data = split_train_test(reduced_emails, train_ratio)
    
    print("Writing to {}...".format(args.outputfile))
    with open(args.outputfile, 'w') as f:
        f.write("type,author,features\n")
    write_to_file(train_data, args.outputfile, "train")
    write_to_file(test_data, args.outputfile, "test")

    print("Done!")
    
