import os
import sys
import argparse
sys.path.append('../BRAD')
from BRAD.rag import create_database

def main():
    parser = argparse.ArgumentParser(description="Create an enrichment database from documents.")

    parser.add_argument(
        "--documents_directory", "-d",
        type=str,
        default="documents",
        help="Path to the directory containing document files."
    )
    parser.add_argument(
        "--database_directory", "-D",
        type=str,
        default="databases",
        help="Path to the directory where the database should be stored."
    )
    parser.add_argument(
        "--database_name", "-n",
        type=str,
        default="enrichment_database",
        help="Name of the database to be created."
    )
    parser.add_argument(
        "--text_size", "-s",
        type=int,
        default=700,
        help="Size of text chunks for processing."
    )
    parser.add_argument(
        "--text_overlap", "-o",
        type=int,
        default=700,
        help="Number of overlapping characters between chunks."
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output."
    )

    args = parser.parse_args()

    create_database(
        docsPath=args.documents_directory,
        dbName=args.database_name,
        dbPath=args.database_directory,
        chunk_size=[args.text_size],
        chunk_overlap=[args.text_overlap],
        v=args.verbose
    )

if __name__ == "__main__":
    main()
