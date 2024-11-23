# Copyright (c) Moitree Basu under MIT License.
# Data processing utilities for document classification
# Code: https://github.com/moitreebasu1990/intellidoc

"""Text processing module for NLP tasks.

This module provides comprehensive text processing functionality including:
    - Tokenization
    - Stop word removal
    - Part-of-speech tagging
    - Lemmatization
    - Named Entity Recognition (NER)

The module processes text data through multiple stages of NLP pipeline and
saves the results at each stage for further analysis or model training.
"""

import argparse

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer

# Download required NLTK resources
nltk.download("averaged_perceptron_tagger_eng")
nltk.download("wordnet")
nltk.download("maxent_ne_chunker_tab")

# Initialize NLP components
lemmatizer = WordNetLemmatizer()
tokenizer = RegexpTokenizer(r"\w+")


def tokenize_text(text: str) -> list[str]:
    """Tokenize text into words using regex pattern.

    Args:
        text (str): Input text to tokenize.

    Returns:
        list[str]: List of tokens extracted from the text.
    """
    return tokenizer.tokenize(text)


def stopwords_removal(tokens: list[str]) -> list[str]:
    """Remove English stop words from a list of tokens.

    Args:
        tokens (list[str]): List of tokens to process.

    Returns:
        list[str]: List of tokens with stop words removed.
    """
    filtered_tokens = [
        word for word in tokens if word not in set(stopwords.words("english"))
    ]
    return filtered_tokens


def pos_tagging(tokens: list[str]) -> list[tuple[str, str]]:
    """Perform part-of-speech tagging on tokens.

    Args:
        tokens (list[str]): List of tokens to tag.

    Returns:
        list[tuple[str, str]]: List of tuples containing (token, POS tag).
    """
    pos_tagged_token = nltk.pos_tag(tokens)
    return pos_tagged_token


def lemmatize_tokens(tokens: list[str]) -> list[str]:
    """Lemmatize tokens to their base form.

    Args:
        tokens (list[str]): List of tokens to lemmatize.

    Returns:
        list[str]: List of lemmatized tokens.
    """
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return lemmatized_tokens


def ner_tagging(tokens: list[tuple[str, str]]) -> nltk.Tree:
    """Perform Named Entity Recognition on POS-tagged tokens.

    Args:
        tokens (list[tuple[str, str]]): List of POS-tagged tokens.

    Returns:
        nltk.Tree: Tree structure containing named entity information.
    """
    ner_tags = nltk.ne_chunk(tokens)
    return ner_tags


def process_data(input_csv: str, output_csv: str) -> None:
    """Process text data through the complete NLP pipeline.

    This function applies the following processing steps:
    1. Load data from CSV
    2. Convert text to lowercase
    3. Tokenize text
    4. Remove stop words
    5. Apply POS tagging
    6. Lemmatize tokens
    7. Perform NER tagging
    8. Save processed data to CSV

    Args:
        input_csv (str): Path to input CSV file containing raw text data.
        output_csv (str): Path where processed data will be saved.
    """
    # Load input data
    df = pd.read_csv(input_csv)

    # Convert text to lowercase
    df["processed_text"] = df["text"].str.lower()

    # Apply NLP pipeline steps
    df["tokens"] = df["processed_text"].apply(tokenize_text)
    df["filtered_tokens"] = df["tokens"].apply(stopwords_removal)
    df["pos_tags"] = df["filtered_tokens"].apply(pos_tagging)
    df["lemmatized_tokens"] = df["filtered_tokens"].apply(lemmatize_tokens)
    df["entities"] = df["pos_tags"].apply(ner_tagging)

    # Save processed data
    df.to_csv(output_csv, index=False)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = argparse.ArgumentParser(
        description="Process text data through NLP pipeline"
    )

    # Define command line arguments
    parser.add_argument(
        "--input_csv",
        type=str,
        default="./data/interim/preprocessed_data.csv",
        help="Path to input CSV file containing raw text",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="./data/processed/final_data.csv",
        help="Path where processed data will be saved",
    )

    # Parse and process arguments
    args = parser.parse_args()
    process_data(args.input_csv, args.output_csv)
