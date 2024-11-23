# Copyright (c) Moitree Basu under MIT License.
# Tests for data preprocessing implementation
# Code: https://github.com/moitreebasu1990/intellidoc

import os

import pandas as pd
import pytest

from intellidoc.data_processing.data_preprocessing import (
    extract_text_from_pdf,
    extract_text_to_csv,
    process_pdf,
)


def test_extract_text_from_pdf(tmp_path):
    # Create a simple PDF file for testing
    pdf_content = "Test content for PDF extraction"
    pdf_path = tmp_path / "test.pdf"

    # Note: Creating a real PDF file would require additional setup
    # For now, we'll mock the PdfReader

    with pytest.raises(Exception):
        # Should raise an exception for non-existent file
        extract_text_from_pdf(str(pdf_path))


def test_process_pdf(tmp_path):
    # Create a test directory structure
    category = "test_category"
    test_dir = tmp_path / category
    test_dir.mkdir()
    pdf_path = test_dir / "test.pdf"

    # Note: Creating a real PDF file would require additional setup
    # For now, we'll test the error case

    with pytest.raises(Exception):
        process_pdf(str(pdf_path))


def test_extract_text_to_csv(tmp_path):
    # Create test directory structure
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    category_dir = input_dir / "category1"
    category_dir.mkdir()

    # Create output path
    output_csv = tmp_path / "output.csv"

    # Test with empty directory
    extract_text_to_csv(str(input_dir), str(output_csv))

    # Check if output CSV was created
    assert os.path.exists(output_csv)
    df = pd.read_csv(output_csv)
    assert len(df) == 0  # Should be empty as no PDFs were processed
