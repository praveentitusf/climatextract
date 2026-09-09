"""Handles extraction of text and tables from PDF pages"""
import csv
import os

import pandas as pd
import torch
from docling.document_converter import DocumentConverter
from pdf2image import convert_from_path
from transformers import DetrImageProcessor, TableTransformerForObjectDetection

from climatextract.semantic_search import Page


class PageTextAndTableExtractor:
    """Handles extraction of text and tables from PDF pages
    Consolidates all table extraction functionality (also methods previously in table_helper.py)
    """

    def __init__(self):
        """Initialize the extractor"""
        pass


    async def extract_text_and_tables_from_pages(self, relevant_pages: list[Page], filename: str) -> tuple[list[str], int]:
        """Extract both text and tables from relevant pages.

        Returns:
            Tuple of (page contents with tables merged in, number of tables extracted).
        """
        relevant_tables = self.get_tables_for_relevant_pages(relevant_pages, filename)
        num_tables = sum(1 for t in relevant_tables if t)
        # If there are relevant pages with tables, pass them to LLM
        if any(inner_list for inner_list in relevant_tables):
            return ([''.join(str(x) for x in relevant_tables[idx]) + ' ' +  page.page_content for idx, page in enumerate(relevant_pages)], num_tables)
        else:
            return ([page.page_content for page in relevant_pages], 0)


    def get_tables_for_relevant_pages(self, relevant_pages, filename):
        """Extract tables from relevant pages and return them with the filtered relevant pages."""
        relevant_page_numbers = [
            page.page_index + 1 for page in relevant_pages]

        extracted_tables = self.extract_tables_from_pages(
            filename, relevant_page_numbers)

        tables = self.match_relevant_pages_and_tables(
            relevant_pages, extracted_tables)

        return tables

    def match_relevant_pages_and_tables(self, relevant_pages, extracted_tables):
        """Match relevant pages with extracted tables."""
        tables_padded = []
        for page in relevant_pages:
            page_number = page.page_index + 1
            if page_number in extracted_tables.keys():
                tables_padded.append(extracted_tables[page_number])
            else:
                tables_padded.append([])
        return tables_padded


    def extract_tables_from_pages(self, file_name: str, rel_page_numbers: list[int]) -> dict[int, list]:
        """
        Extract tables from the specified pages of the PDF file,
        either by reading in already extracted tables or by extracting them from the PDF file,
        checking if the page contains a table first.
        """
        tables = {}
        for page in rel_page_numbers:
            report_name = file_name.split('/')[-1].replace('.pdf', '')
            path_to_table_cells = f'./data/processed/tables/{report_name}_{page}_table_cells.csv'
            # In case table has already been extracted from the page
            if os.path.exists(path_to_table_cells):
                with open(path_to_table_cells, encoding='utf-8') as f:
                    table = list(csv.reader(f, delimiter=","))
                tables[page] = table
            # In case table has not been extracted from the page
            else:
                # Check whether the page contains a table
                if self.check_if_table_on_page(file_name, page):
                    table_extracted = self.extract_table_from_page(file_name, page)
                    table_cleaned = self.header_value_pairs(table_extracted)
                    self.save_table(path_to_table_cells, table_cleaned)
                    tables[page] = table_cleaned

        return tables


    def check_if_table_on_page(self, file_name: str, relevant_page: int) -> bool:
        '''Check if the page contains a table'''
        processor = DetrImageProcessor.from_pretrained(
            "microsoft/table-transformer-detection")
        model = TableTransformerForObjectDetection.from_pretrained(
            "microsoft/table-transformer-detection")

        image = convert_from_path(
            file_name, first_page=relevant_page, last_page=relevant_page, fmt='jpeg')[0]
        image = image.convert("RGB")
        width, height = image.size
        image.resize((int(width*0.5), int(height*0.5)))

        inputs = processor(images=image, return_tensors="pt")

        # Perform inference
        with torch.no_grad():
            outputs = model(**inputs)

        # Identify the tables in the image of pdf page
        results = processor.post_process_object_detection(
            outputs, threshold=0.7, target_sizes=[(height, width)])[0]
        if len(results['scores']) != 0 and any(t > 0.98 for t in results['scores']):
            return True
        else:
            return False


    def extract_table_from_page(self, file_name: str, page: int) -> list[pd.DataFrame]:
        """Extracts every table in the document, converting the file once."""
        converter = DocumentConverter()
        result = converter.convert(source=file_name, page_range=(page, page))
        doc = result.document
        dfs = [table.export_to_dataframe(doc=doc) for table in doc.tables]

        return dfs


    def header_value_pairs(self, dfs: list) -> list[tuple]:
        """Converts each DataFrame's rows to tuples."""
        result = []
        for df in dfs:
            for row in df.to_dict(orient="records"):
                result.append(
                    tuple(f"{col}: {value}" for col, value in row.items() if pd.notnull(value) and value != "")
                )
        return result


    def save_table(self, path_to_table_cells: str, table: list):
        """Save the table to a CSV file"""
        os.makedirs(os.path.dirname(path_to_table_cells), exist_ok=True)
        with open(path_to_table_cells, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(table)
        return
