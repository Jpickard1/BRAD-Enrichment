import os
import sys
import click
import argparse

original_stdout, original_stderr = sys.stdout, sys.stderr
null_output = open(os.devnull, 'w')
sys.stdout, sys.stderr = null_output, null_output
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFLoader
sys.stdout, sys.stderr = original_stdout, original_stderr


def build_database(documents_directory, database_directory, database_name, text_size, text_overlap, verbose):
    """
    Builds a LangChain Chroma database from documents in the specified directory.
    """
    # Load documents
    loader = DirectoryLoader(documents_directory,
                             glob="**/*.pdf",
                             loader_cls=PyPDFLoader, 
                             # loader_kwargs=text_loader_kwargs,
                             show_progress=True,
                             use_multithreading=True)
    documents = loader.load()
    
    if verbose:
        print(f"Loaded {len(documents)} documents from {documents_directory}")
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=text_size,
        chunk_overlap=text_overlap
    )
    docs = text_splitter.split_documents(documents)
    
    if verbose:
        print(f"Split into {len(docs)} chunks")
    
    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings()
    
    # Create Chroma database
    db_path = os.path.join(database_directory, database_name)
    vector_db = Chroma.from_documents(docs, embeddings, persist_directory=db_path)
    vector_db.persist()
    
    if verbose:
        print(f"Database successfully saved to {db_path}")

def load_database(database_directory='databases', database_name='enrichment_database', db_path=None, verbose=True):
    """
    Loads an existing Chroma database and prints document and chunk counts.
    """
    if db_path is None:
        db_path = os.path.join(database_directory, database_name)
        
    vector_db = Chroma(persist_directory=db_path, embedding_function=HuggingFaceEmbeddings())
    
    documents_count = len(vector_db.get()['documents'])
    chunks_count = len(vector_db.get()['ids'])
    
    if verbose:
        print(f"Loaded database from {db_path}")
        print(f"Number of documents: {documents_count}")
        print(f"Number of chunks: {chunks_count}")
    
    return vector_db

def main():
    parser = argparse.ArgumentParser(description="Create an enrichment database from documents.")

    parser.add_argument("--documents_directory", "-d", type=str, default="documents", help="Path to the directory containing document files.")
    parser.add_argument("--database_directory", "-D", type=str, default="databases", help="Path to the directory where the database should be stored.")
    parser.add_argument("--database_name", "-n", type=str, default="enrichment_database", help="Name of the database to be created.")
    parser.add_argument("--text_size", "-s", type=int, default=700, help="Size of text chunks for processing.")
    parser.add_argument("--text_overlap", "-o", type=int, default=100, help="Number of overlapping characters between chunks.")
    parser.add_argument("--verbose", "-v", default="store_true", help="Enable verbose output.")

    args = parser.parse_args()

    # Create the database
    build_database(
        args.documents_directory,
        args.database_directory,
        args.database_name,
        args.text_size,
        args.text_overlap,
        args.verbose
    )

    if args.verbose:
        print("Database creation complete.")

@click.command()
@click.option("--documents-directory", "-d", type=str, default="documents", help="Path to the directory containing document files.")
@click.option("--database-directory", "-D", type=str, default="databases", help="Path to the directory where the database should be stored.")
@click.option("--database-name", "-n", type=str, default="enrichment_database", help="Name of the database to be created.")
@click.option("--text-size", "-s", type=int, default=700, help="Size of text chunks for processing.")
@click.option("--text-overlap", "-o", type=int, default=100, help="Number of overlapping characters between chunks.")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output.")
def cli(documents_directory, database_directory, database_name, text_size, text_overlap, verbose):
    """CLI wrapper for building a LangChain Chroma database from documents."""
    build_database(documents_directory, database_directory, database_name, text_size, text_overlap, verbose)

    if verbose:
        print("Database creation complete.")

if __name__ == "__main__":
    main()
