import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment
import argparse

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import gget
import sys
import os
sys.path.append('../BRAD')
from rich import print
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# Imports for BRAD library
from BRAD.agent import Agent # , AgentFactory

# For the Video RAG
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

VERBOSE = True

THRESHOLD_P_VALUE = 0.05
MAXIMUM_ENRICHMENT_TERMS = 10
MINIMUM_ENRICHMENT_TERMS = 3

EMBEDDINGS_MODEL = HuggingFaceEmbeddings(model_name='BAAI/bge-base-en-v1.5')

DATABASE_FOLDER = None

SYSTEM_ENRICHMENT_PROMPT = """You are a bioinformatics specialist tasked with summarizing gene enrichment results.
Provide a clear, concise, and scientifically accurate 1-5 sentence explanation of the biological significance of the
enrichment terms, pathways, cell types, or other information. Focus solely on the function, relevance, process, and
biological significance of these enrichment terms. These terms are found from the database: {database_name}.

Do not include information about yourself or the nature of this request.

Enrichment term: {enrichment_term}"""
# Associated Genes: {gene_list}"""

SYSTEM_ENRICHMENT_TYPE_PROMPT = """You are a bioinformatics specialist tasked with summarizing gene enrichment results.
Provide a clear, concise, and scientifically accurate 2-5 sentence explanation of the following enrichment
dataframe. Focus primarily on the relationship between the pathways with one another and on the high level themes
observed accross the enrichment results. Include information such as what biological processes, pathways, celltypes,
diseases, information and biological significance that appears accross multiple enrichment results, or if there is diversity
accross the results indicate that.

Do not include information about yourself or the nature of this request.

Enrichment Results: """

SYSTEM_OVERVIEW_PROMPT = """You are a bioinformatics specialist tasked with summarizing gene enrichment results. Several enrichment 
databases have been used and summarized. Your task is to provide a clear, concise, and scientifically accurate 2-5 sentence 
explanation and synthesis of the results from the different enrichment databases. Focus primarily on the relationship between 
the biological significance and themes that are observed accross multiple enrichment databases. 

Include information such as what biological processes, pathways, celltypes, diseases, ontology terms, information and biological 
significance appear accross multiple enrichment results, or if there is diversity accross the results indicate that. Do not repeat 
the information from each enrichment result, but instead focus on a synthesis of this information. 

Do not include information about yourself or the nature of this request.

Enrichment Results: """

def process_pathway(pathway_database_pair):
    """Initialize a separate BRAD instance for each pathway and invoke it."""
    pathway, database, literature_database = pathway_database_pair
    brad = Agent(
        tools=['RAG'],
        gui=True,
        config='config.json',
        interactive=False
    )
    print(f"{literature_database=}")
    # print(isinstance(DATABASE_FOLDER, None))
    print(f"{(literature_database is not None)=}")
    if literature_database is not None:
        try:
            persist_directory = literature_database # os.path.join(DATABASE_FOLDER)
            print(f"{persist_directory=}")
            vectordb = Chroma(
                persist_directory=persist_directory,  # Path where the database was saved
                embedding_function=EMBEDDINGS_MODEL,  # Embedding model
                collection_name="default"  # Ensure this matches what was used during writing
            )
            print(f"{vectordb=}")
            brad.state['databases']['RAG'] = vectordb
            print(f"{brad.state['databases']['RAG']=}")
            print("Database connected!")
            print(f"{vectordb.get().keys()=}")
            print(f"{len(vectordb.get()['ids'])=}")
        except Exception as e:
            print("Database NOT connected!")
            print(f"{vectordb.get().keys()=}")
            print(f"{len(vectordb.get()[vectordb.get().keys()[0]])=}")
            logger.warning(f"Failed to initialize ChromaDB: {e}")
    print(f"{brad.state=}")
    return brad.invoke(SYSTEM_ENRICHMENT_PROMPT.format(database_name=database, enrichment_term=pathway))

def summarize_enrichment_type(enrichment_df):
    """Summarize the results of an enrichment dataframe"""
    brad = Agent(
        tools=['RAG'],
        gui=True,
        config='config.json',
        interactive=False
    )
    minimal_enrichment_df = enrichment_df[['rank', 'path_name', 'BRAD Result']].copy()
    return brad.invoke(SYSTEM_ENRICHMENT_TYPE_PROMPT + minimal_enrichment_df.to_json(orient="records"))

def perform_enrichment(
        gene_string,
        databases = ['KEGG_2021_Human', 'GO_Biological_Process_2021', 'PanglaoDB_Augmented_2021'], 
        threshold_p_value=THRESHOLD_P_VALUE,
        minimum_enrichment_terms = MINIMUM_ENRICHMENT_TERMS,
        maximum_enrichment_terms = MAXIMUM_ENRICHMENT_TERMS,
        literature_database=DATABASE_FOLDER,
        verbose=VERBOSE
    ):
    print(f"{gene_string=}") if verbose else None
    genes = [gene.strip() for gene in gene_string.split(',')]
    print(f"{genes=}") if verbose else None
    output_file = "enrichment_results.xlsx"

    # Perform enrichment
    enrichment_dfs = []
    for db in databases:
        df = gget.enrichr(genes, database=db)
        enrichment_dfs.append(df)

    # Preprocess the dataframes
    for dfi, df in enumerate(enrichment_dfs):
        filtered_df = df[df['p_val'] < threshold_p_value].copy()
        if filtered_df.shape[0] < minimum_enrichment_terms:
            filtered_df = df.iloc[:min(minimum_enrichment_terms, df.shape[0])]
        filtered_df = filtered_df.sort_values(by='p_val').head(min(filtered_df.shape[0], maximum_enrichment_terms))
        filtered_df = filtered_df.drop(["database"], axis=1)
        enrichment_dfs[dfi] = filtered_df.copy()

    # Process Enrichment with RAG
    for dfi, df in enumerate(enrichment_dfs):
        pathways = df['path_name'].tolist()
        pathway_database_pairs = [(pw, databases[dfi], literature_database) for pw in pathways]
        
        # Use ThreadPoolExecutor for parallel execution
        with ThreadPoolExecutor() as executor:
            enrichment_summaries = list(executor.map(process_pathway, pathway_database_pairs))

        df.loc[:, 'BRAD Result'] = enrichment_summaries  # Safe assignment
        enrichment_dfs[dfi] = df.copy()  # Update the list with the processed DataFrame


    # Use ThreadPoolExecutor for parallel execution
    with ThreadPoolExecutor() as executor:
        enrichment_summaries = list(executor.map(summarize_enrichment_type, enrichment_dfs))

    highlevel_df = pd.DataFrame(
        {
            "Enrichment Database": databases,
            "Results": enrichment_summaries
        }
    )
    brad = Agent(
        tools=['RAG'],
        gui=True,
        config='config.json',
        interactive=False
    )
    enrichment_overview = brad.invoke(SYSTEM_ENRICHMENT_TYPE_PROMPT + highlevel_df.to_json(orient="records"))
    databases.insert(0, "Overview")
    enrichment_summaries.insert(0, enrichment_overview)
    highlevel_df = pd.DataFrame(
        {
            "Enrichment Database": databases,
            "Results": enrichment_summaries
        }
    )
    highlevel_df.to_csv('highlevel_df.csv')
    reference_df = pd.DataFrame(
        {
            'Topic': [
                'Gene List',
                'Fisher\'s Exact Test', 
                'Gene Ontology (GO)', 
                'Panglaodb Database'
            ],
            'Description': [
                str(genes),
                "Fisher's Exact Test is a statistical test used to determine if there are nonrandom associations between two categorical variables. In gene enrichment, it is often used to evaluate whether a particular gene is overrepresented in a predefined set of categories (e.g., pathways or GO terms) compared to the whole genome.",
                "Gene Ontology (GO) provides a standardized vocabulary of terms to describe gene functions, cellular components, and biological processes. GO terms help in annotating genes and interpreting gene function across different species.",
                "Panglaodb is a comprehensive database that provides gene expression data from multiple species, integrating data from various sources. It allows researchers to access transcriptomic and functional genomics data to support gene enrichment analyses."
            ]
        }
    )

    # Save to Excel with two sheets
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        # Write DataFrames to sheets
        highlevel_df.to_excel(writer, sheet_name="Overview", index=False)
        for dfi, df in enumerate(enrichment_dfs):
            df.to_excel(writer, sheet_name=databases[dfi+1], index=False)
        reference_df.to_excel(writer, sheet_name="Reproducibility", index=False)

        # Access the workbook
        workbook = writer.book

        # Apply formatting to each sheet
        for sheeti, sheet_name in enumerate(writer.sheets):
            worksheet = writer.sheets[sheet_name]
            worksheet.freeze_panes = "A2"

            for coli, col in enumerate(worksheet.iter_cols()):
                col_name = col[0].value  # First row contains column names

                # Adjust column widths for specific columns
                if col_name == "BRAD Result":
                    worksheet.column_dimensions[col[0].column_letter].width = (worksheet.column_dimensions[col[0].column_letter].width or 8.43) * 6
                elif col_name == "rank":
                    worksheet.column_dimensions[col[0].column_letter].width = (worksheet.column_dimensions[col[0].column_letter].width or 8.43) * 0.5
                elif col_name in ["path_name", 'combined_score', 'overlapping_genes']:
                    worksheet.column_dimensions[col[0].column_letter].width = (worksheet.column_dimensions[col[0].column_letter].width or 8.43) * 1.5
                elif col_name in ["Description", "Results"]:
                    worksheet.column_dimensions[col[0].column_letter].width = (worksheet.column_dimensions[col[0].column_letter].width or 8.43) * 6
                elif col_name in ["Topic", "Enrichment Database"]:
                    worksheet.column_dimensions[col[0].column_letter].width = (worksheet.column_dimensions[col[0].column_letter].width or 8.43) * 3

                # Ensure wrapping text in the first sheet
                if sheeti == 0:  # Only apply to "Overview" sheet
                    for cell in col:
                        cell.alignment = Alignment(wrap_text=True, vertical="top")

                # Apply text wrapping for specific columns in other sheets
                if col_name in ["BRAD Result", "path_name", "Description", "Enrichment Database", "Results"]:
                    for cell in col[1:]:  # Skip header row
                        cell.alignment = Alignment(wrap_text=True, vertical="top")
                else:
                    for cell in col[1:]:  # Skip header row
                        cell.alignment = Alignment(vertical="top")

        workbook.save(output_file)

def main():
    parser = argparse.ArgumentParser(description="Perform gene enrichment analysis using BRAD.")

    # Required argument: gene list
    parser.add_argument(
        "gene_string",
        type=str,
        help="Comma-separated list of genes (e.g., MYOD, P53, CDK2)."
    )

    # Optional arguments
    parser.add_argument(
        "--databases",
        type=str,
        nargs="+",
        default=['KEGG_2021_Human', 'GO_Biological_Process_2021', 'PanglaoDB_Augmented_2021'],
        help="List of databases to use for enrichment (default: KEGG, GO, PanglaoDB)."
    )

    parser.add_argument(
        "--threshold_p_value",
        type=float,
        default=0.05,
        help="P-value threshold for enrichment results (default: 0.05)."
    )

    parser.add_argument(
        "--minimum_enrichment_terms",
        type=int,
        default=3,
        help="Minimum number of enrichment terms to report (default: 10)."
    )

    parser.add_argument(
        "--maximum_enrichment_terms",
        type=int,
        default=10,
        help="Maximum number of enrichment terms to report (default: 500)."
    )

    parser.add_argument(
        "--literature_database",
        type=str,
        default="databases/enrichment_database"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose mode for debugging and logging."
    )

    # Parse arguments
    args = parser.parse_args()
    
    DATABASE_FOLDER = args.literature_database
    print(f"{DATABASE_FOLDER=}")

    # Run the enrichment function
    results = perform_enrichment(
        gene_string=args.gene_string,
        databases=args.databases,
        threshold_p_value=args.threshold_p_value,
        minimum_enrichment_terms=args.minimum_enrichment_terms,
        maximum_enrichment_terms=args.maximum_enrichment_terms,
        literature_database=args.literature_database,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()
