# Copyright (c) Moitree Basu under MIT License.
# Data preprocessing utilities for document classification
# Code: https://github.com/moitreebasu1990/intellidoc

"""PDF text extraction and preprocessing module.

This module provides functionality for extracting text from PDF files and preprocessing
it for machine learning tasks. Features include:
    - Parallel processing of PDF files
    - Progress tracking with tqdm
    - Automatic label extraction from directory structure
    - CSV output generation

The module assumes PDFs are organized in directories where the directory name
represents the label/category of the PDFs within it.
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
from PyPDF2 import PdfReader
from tqdm import tqdm


def extract_text_from_pdf(file_path: str) -> str:
    """Extract text content from a PDF file.

    Args:
        file_path (str): Path to the PDF file.

    Returns:
        str: Extracted text content from all pages of the PDF.
    """
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text


def process_pdf(file_path: str) -> tuple[str, str]:
    """Process a single PDF file and extract its text and label.

    The label is derived from the name of the directory containing the PDF.

    Args:
        file_path (str): Path to the PDF file.

    Returns:
        tuple[str, str]: A tuple containing:
            - content (str): Extracted text from the PDF
            - label (str): Label derived from the parent directory name
    """
    content = extract_text_from_pdf(file_path)
    label = os.path.basename(os.path.dirname(file_path))
    return content, label


def extract_text_to_csv(input_folder: str, output_csv: str) -> None:
    """Extract text from all PDFs in a folder and save to CSV.

    This function:
    1. Recursively finds all PDF files in the input folder
    2. Processes them in parallel using ProcessPoolExecutor
    3. Extracts text content and labels
    4. Saves the results to a CSV file

    Args:
        input_folder (str): Path to the folder containing PDF files organized in
            labeled directories.
        output_csv (str): Path where the output CSV file will be saved.
    """
    # Initialize lists for extracted data
    text = []
    label = []

    # Find all PDF files recursively
    pdf_files = [
        os.path.join(root, filename)
        for root, _, files in os.walk(input_folder)
        for filename in files
        if filename.endswith(".pdf")
    ]

    # Set up progress tracking
    progress_bar = tqdm(total=len(pdf_files), desc="Processing PDFs")

    # Process PDFs in parallel
    with ProcessPoolExecutor() as executor:
        # Submit all PDF files for processing
        futures = {
            executor.submit(process_pdf, file_path): file_path
            for file_path in pdf_files
        }

        # Collect results as they complete
        for future in as_completed(futures):
            content, lbl = future.result()
            text.append(content)
            label.append(lbl)
            progress_bar.update(1)

    progress_bar.close()

    # Create and save DataFrame
    df = pd.DataFrame({"text": text, "label": label})
    df.to_csv(output_csv, index=False)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = argparse.ArgumentParser(
        description="Extract text from PDFs and save to CSV"
    )

    # Define command line arguments
    parser.add_argument(
        "--input_folder",
        type=str,
        default="./data/raw/",
        help="Path to folder containing PDFs organized in labeled directories",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="./data/interim/preprocessed_data.csv",
        help="Path where the output CSV file will be saved",
    )

    # Parse and process arguments
    args = parser.parse_args()
    extract_text_to_csv(args.input_folder, args.output_csv)
