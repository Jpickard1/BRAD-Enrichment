### Import statements
import os
import re
import sys
import click
import logging
import argparse
import warnings
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.WARNING)  # Set to WARNING to reduce noise from imported packages
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)  # Set our own logger to INFO

import gget
logging.getLogger("gget.utils").setLevel(logging.WARNING)

# Set up progress bar
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    SpinnerColumn
)
PROGRESS_COLUMNS = [
    SpinnerColumn(),
    TextColumn("[bold blue]{task.description}"),
    BarColumn(bar_width=60),
    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    TimeElapsedColumn(),
    TimeRemainingColumn(),
]

progress = Progress(*PROGRESS_COLUMNS)
overall_task = None


# Import BRAD agent and rag database with surpressed warning messages
null_output = open(os.devnull, 'w')
original_stdout, original_stderr = sys.stdout, sys.stderr
sys.stdout, sys.stderr = null_output, null_output
from brad_enrichment.enrichment_literature_database import load_database
warnings.filterwarnings("ignore", category=UserWarning, module="langchain")
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_core")
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_community")
from langchain._api import LangChainDeprecationWarning
warnings.simplefilter("ignore", category=LangChainDeprecationWarning)
warnings.simplefilter("ignore")
from BRAD.agent import Agent
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
EMBEDDINGS_MODEL = HuggingFaceEmbeddings(model_name='BAAI/bge-base-en-v1.5')
sys.stdout, sys.stderr = original_stdout, original_stderr


### Set Globals
original_stdout, original_stderr = sys.stdout, sys.stderr
EMBEDDINGS_MODEL = None
VERBOSE = True
THRESHOLD_P_VALUE = 0.05
MAXIMUM_ENRICHMENT_TERMS = 10
MINIMUM_ENRICHMENT_TERMS = 3
# CONFIG_PATH=os.path.join("..", "config.json")
CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config.json"))

DATABASE_FOLDER = None

DEFAULT_QUERY = "No additional requests. Please summarize the information."

SYSTEM_ENRICHMENT_PROMPT = """**Instructions**: You are a bioinformatics specialist tasked with summarizing gene enrichment results.
Provide a clear, concise, and scientifically accurate 1-5 sentence explanation of the biological significance of the
enrichment terms, pathways, cell types, or other information. Focus solely on the function, relevance, process, and
biological significance of these enrichment terms. These terms are found from the database: {database_name}.

Do not include information about yourself or the nature of this request.

**Enrichment term**: {enrichment_term}"""
# Associated Genes: {gene_list}"""

SYSTEM_ENRICHMENT_TYPE_PROMPT = """**Instructions**: You are a bioinformatics specialist tasked with summarizing gene enrichment results.
Provide a clear, concise, and scientifically accurate 2-5 sentence explanation of the following enrichment
dataframe. Focus primarily on the relationship between the pathways with one another and on the high level themes
observed accross the enrichment results. Include information such as what biological processes, pathways, celltypes,
diseases, information and biological significance that appears accross multiple enrichment results, or if there is diversity
accross the results indicate that. Additionally, please respond to any specific requests in the user query.

**User query**: {human_input}

Do not include information about yourself or the nature of this request.

**Enrichment Database**: {database}

**Enrichment Results**: """

SYSTEM_OVERVIEW_PROMPT = """**Instructions**: You are a bioinformatics specialist tasked with summarizing gene enrichment results. Several enrichment 
databases have been used and summarized. Your task is to provide a clear, concise, and scientifically accurate 2-5 sentence 
explanation and synthesis of the results from the different enrichment databases. Focus primarily on the relationship between 
the biological significance and themes that are observed accross multiple enrichment databases. Additionally, please respond
to any specific requests in the user query.

**User query**: {human_input}

Include information such as what biological processes, pathways, celltypes, diseases, ontology terms, information and biological 
significance appear accross multiple enrichment results, or if there is diversity accross the results indicate that. Do not repeat 
the information from each enrichment result, but instead focus on a synthesis of this information. 

Do not include information about yourself or the nature of this request.

**Enrichment Results**: """

def process_pathway(pathway_database_pair):
    """Initialize a separate BRAD instance for each pathway and invoke it."""
    pathway, database, literature_database, sources_only = pathway_database_pair
    brad = Agent(
        tools=['RAG'],
        gui=True,
        config=CONFIG_PATH,
        interactive=False
    )
    rag = False
    if literature_database is not None:
        try:
            vectordb = load_database(db_path=literature_database, verbose=False)
            brad.state['databases']['RAG'] = vectordb
            rag = True
        except Exception as e:
            logger.warning(f"Failed to initialize ChromaDB: {e}")
    sources, ragtext = [], []
    if not sources_only:
        response = brad.invoke(SYSTEM_ENRICHMENT_PROMPT.format(database_name=database, enrichment_term=pathway))
    else:
        response = "Sources Only"
    if rag:
        for _, doc in enumerate(brad.state['process']['steps'][0]['docs-to-gui']):
            sources.append(doc["source"])
            ragtext.append(doc["text"])
    cost = brad.state['process']['steps'][-1]['api-info']['Total Cost (USD)']
    return response, sources, ragtext, cost

def summarize_enrichment_type(enrichment_df):
    """Summarize the results of an enrichment dataframe"""
    enrichment_df, database, literature_database, sources_only, query = enrichment_df
    brad = Agent(
        tools=['RAG'],
        gui=True,
        config=CONFIG_PATH,
        interactive=False
    )
    print(f"{literature_database=}")
    print(f"{(literature_database is not None)=}")
    rag = False
    if literature_database is not None:
        try:
            vectordb = load_database(db_path=literature_database, verbose=False)
            brad.state['databases']['RAG'] = vectordb
            rag = True
        except Exception as e:
            logger.warning(f"Failed to initialize ChromaDB: {e}")
    minimal_enrichment_df = enrichment_df[['rank', 'path_name', 'Summary']].copy()
    sources, ragtext = [], []
    if not sources_only:
        response = brad.invoke(SYSTEM_ENRICHMENT_TYPE_PROMPT.format(database=database, human_input=query) + minimal_enrichment_df.to_json(orient="records"))
    else:
        response = "Sources Only"
    if rag:
        for _, doc in enumerate(brad.state['process']['steps'][0]['docs-to-gui']):
            sources.append(doc["source"])
            ragtext.append(doc["text"])
    cost = brad.state['process']['steps'][-1]['api-info']['Total Cost (USD)']
    return response, sources, ragtext, cost

def build_reproducibility_report(gene_list=None, query=None, model_name=None, temperature=None, cost=None):
    """
    Generate a reproducibility report as a Pandas DataFrame containing references  
    to key resources related to BRAD and the provided gene list.

    The report includes:
    - A hyperlink to the BRAD repository.
    - A hyperlink to the related research paper.
    - A section for a user-provided gene list.
    - A section for a user-provided query (if different from DEFAULT_QUERY).

    Parameters:
    -----------
    gene_list : str, optional
        A comma-separated string of gene names to include in the report.  
        Defaults to None.

    query : str, optional
        A user-provided query. If different from DEFAULT_QUERY, it will be included in the report.

    Returns:
    --------
    pd.DataFrame
        A DataFrame containing the report, with clickable hyperlinks for reference links.
    """
    COLUMN_ONE = "Reproducibility Report"
    COLUMN_TWO = ""
    reference_df = pd.DataFrame(
        {
            COLUMN_ONE: [
                'BRAD Agent',
                'Paper',
                '',
                'Gene List',
                'Enrichment Databases',
                '',
            ],
            COLUMN_TWO: [
                '=HYPERLINK("https://github.com/Jpickard1/BRAD", "Software Repository")',
                '=HYPERLINK("https://arxiv.org/abs/2409.02864", "Language Model Powered Digital Biology with BRAD")',
                '',
                str(gene_list),
                '=HYPERLINK("https://maayanlab.cloud/Enrichr/", "Enrichr")',
                '',
            ]
        }
    )

    # Append the query to the DataFrame if it's different from the default
    if query and query != DEFAULT_QUERY:
        query_entry = pd.DataFrame({COLUMN_ONE: ['Human Query'], COLUMN_TWO: [query]})
        reference_df = pd.concat([reference_df, query_entry], ignore_index=True)

    if model_name is not None:
        model_entry = pd.DataFrame({COLUMN_ONE: ['LLM'], COLUMN_TWO: [model_name]})
        reference_df = pd.concat([reference_df, model_entry], ignore_index=True)

    if temperature is not None:
        temp_entry = pd.DataFrame({COLUMN_ONE: ['Temperature'], COLUMN_TWO: [temperature]})
        reference_df = pd.concat([reference_df, temp_entry], ignore_index=True)

    if cost is not None:
        cost_entry = pd.DataFrame({COLUMN_ONE: ['Cost ($USD)'], COLUMN_TWO: [cost]})
        reference_df = pd.concat([reference_df, cost_entry], ignore_index=True)

    return reference_df


def build_report_file(output_file, highlevel_df, enrichment_dfs, databases, gene_list=None, query=None, model_name=None, temperature=None, cost=None):
    reference_df = build_reproducibility_report(gene_list=gene_list, query=query, model_name=model_name, temperature=temperature, cost=cost)
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
                col_letter = col[0].column_letter

                # Adjust column widths for specific columns
                if col_name in ["Source", "Text", "Reference"]:
                    worksheet.column_dimensions[col_letter].hidden = True
                elif col_name == "Summary":
                    worksheet.column_dimensions[col[0].column_letter].width = (worksheet.column_dimensions[col[0].column_letter].width or 8.43) * 6
                elif col_name == "rank":
                    worksheet.column_dimensions[col[0].column_letter].width = (worksheet.column_dimensions[col[0].column_letter].width or 8.43) * 0.5
                elif col_name in ["path_name", 'combined_score', 'overlapping_genes']:
                    worksheet.column_dimensions[col[0].column_letter].width = (worksheet.column_dimensions[col[0].column_letter].width or 8.43) * 1.5
                elif col_name in ["Description", "Results"]:
                    worksheet.column_dimensions[col[0].column_letter].width = (worksheet.column_dimensions[col[0].column_letter].width or 8.43) * 6
                elif col_name in ["Topic", "Enrichment Database", "Reproducibility Report"]:
                    worksheet.column_dimensions[col[0].column_letter].width = (worksheet.column_dimensions[col[0].column_letter].width or 8.43) * 3

                # Ensure wrapping text in the first sheet
                if sheeti == 0:  # Only apply to "Overview" sheet
                    for cell in col:
                        cell.alignment = Alignment(wrap_text=True, vertical="top")

                # Apply text wrapping for specific columns in other sheets
                if col_name in ["Summary", "path_name", "Description", "Enrichment Database", "Results"]:
                    for cell in col[1:]:  # Skip header row
                        cell.alignment = Alignment(wrap_text=True, vertical="top")
                else:
                    for cell in col[1:]:  # Skip header row
                        cell.alignment = Alignment(vertical="top")

        workbook.save(output_file)
    

def perform_enrichment(
        gene_string,
        databases = ['KEGG_2021_Human', 'GO_Biological_Process_2021', 'PanglaoDB_Augmented_2021'], 
        threshold_p_value=THRESHOLD_P_VALUE,
        minimum_enrichment_terms = MINIMUM_ENRICHMENT_TERMS,
        maximum_enrichment_terms = MAXIMUM_ENRICHMENT_TERMS,
        literature_database=DATABASE_FOLDER,
        output_file="enrichment_results.xlsx",
        query=None,
        sources_only=False, # If true, the only retrieval is performed. If false, the LLM generation occurs as well
        verbose=VERBOSE
    ):
    global overall_task, original_stdout, original_stderr
    null_output = open(os.devnull, 'w')
    
    progress.update(overall_task, advance=5, description="Setting up...")
#    heavy_imports()
    total_cost = 0.0
    
    # 1. split string into a list of genes
    genes = [gene.strip() for gene in gene_string.split(',')]
    progress.update(overall_task, advance=25, description="Preparing gene list...")

    # 2. perform enrichment
    enrichment_dfs = []
    progress_per_db = 15 / len(databases)

    for db_idx, db in enumerate(databases):
        progress.update(overall_task, description=f"Enriching genes against {db}...")
        df = gget.enrichr(genes, database=db)
        enrichment_dfs.append(df)
        progress.update(overall_task, advance=progress_per_db)

    # 3. preprocess the dataframes
    progress.update(overall_task, description="Preprocessing enrichment results...")
    for dfi, df in enumerate(enrichment_dfs):
        filtered_df = df[df['p_val'] < threshold_p_value].copy()
        if filtered_df.shape[0] < minimum_enrichment_terms:
            filtered_df = df.iloc[:min(minimum_enrichment_terms, df.shape[0])]
        filtered_df = filtered_df.sort_values(by='p_val').head(min(filtered_df.shape[0], maximum_enrichment_terms))
        filtered_df['num_genes'] = filtered_df['overlapping_genes'].apply(len)
        filtered_df = filtered_df.drop(["database"], axis=1)
        enrichment_dfs[dfi] = filtered_df.copy()
    progress.update(overall_task, advance=5)
    print("preprocessing complete")
    
    # 4. process enrichment results with RAG
    progress.update(overall_task, description="Processing enrichment pathways...")
    progress_per_df = 30 / len(enrichment_dfs)  # Allocate 30% of progress to RAG analysis

    for dfi, df in enumerate(enrichment_dfs):
#        print(f"{dfi=}")
        # Format the list of pathways to enrich
        pathways = df['path_name'].tolist()
        pathway_database_pairs = [(pw, databases[dfi], literature_database, sources_only) for pw in pathways]
#        print("pathway_database_pairs set")
        progress.update(overall_task, description=f"Analyzing pathways for {databases[dfi]}...")
#        print("progress.update set")
#        print(f"{null_output=}")
        # ThreadPoolExecutor for parallel rag analysis
        sys.stdout, sys.stderr = null_output, null_output
#        print("sys.stdout, sys.stderr set")
        with ThreadPoolExecutor() as executor:
#            print("ThreadPoolExecutor context open")
            enrichment_summaries = list(executor.map(process_pathway, pathway_database_pairs))
#            print("ThreadPoolExecutor context open")

        sys.stdout, sys.stderr = original_stdout, original_stderr
#        print("enrichment_summaries set")

        # Convert BRAD Agent's rag analysis into a dataframe
        enrichment_summaries_pivoted = tuple(map(list, zip(*enrichment_summaries)))
        references = enrichment_summaries_pivoted[2]
        df.loc[:, 'Summary'] = enrichment_summaries_pivoted[0]
        df.loc[:, 'Source'] = enrichment_summaries_pivoted[1]
        df.loc[:, 'Text'] = references
        enrichment_dfs[dfi] = df.copy()  # Update the list with the processed DataFrame
        costs = enrichment_summaries_pivoted[3]
        for c in costs: total_cost += c
        progress.update(overall_task, advance=progress_per_df)

    # 5. Interpret the results of each enrichment database
    progress.update(overall_task, description="Interpreting enrichment database results...")
    pathway_database_pairs = [(enrichment_dfs[dfi], databases[dfi], literature_database, sources_only, query) for dfi in range(len(enrichment_dfs))]

    sys.stdout, sys.stderr = null_output, null_output
    with ThreadPoolExecutor() as executor:
        enrichment_summaries = list(executor.map(summarize_enrichment_type, pathway_database_pairs))
    sys.stdout, sys.stderr = original_stdout, original_stderr
    
    if VERBOSE:
        logger.info("Enrichment summaries generated successfully")
    
    progress.update(overall_task, advance=20)

    # 6. Construct high level enrichment interpretation
    progress.update(overall_task, description="Building high-level interpretation...")
    enrichment_summaries_pivoted = tuple(map(list, zip(*enrichment_summaries)))
    costs = enrichment_summaries_pivoted[3]
    for c in costs: total_cost += c
    highlevel_df = pd.DataFrame(
        {
            "Enrichment Database": pd.Series(databases),
            "Results": pd.Series(enrichment_summaries_pivoted[0]),
            "Text": pd.Series(enrichment_summaries_pivoted[2]),
            "Source": pd.Series(enrichment_summaries_pivoted[1]),
        }
    )
    sys.stdout, sys.stderr = null_output, null_output
    brad = Agent(
        tools=['RAG'],
        gui=True,
        config=CONFIG_PATH,
        interactive=False
    )
    sys.stdout, sys.stderr = original_stdout, original_stderr

    if not sources_only:
        sys.stdout, sys.stderr = null_output, null_output
        enrichment_overview = brad.invoke(SYSTEM_OVERVIEW_PROMPT.format(human_input=query) + highlevel_df.to_json(orient="records"))
        sys.stdout, sys.stderr = original_stdout, original_stderr
    else:
        enrichment_overview = "Sources Only"
    total_cost += brad.state['process']['steps'][-1]['api-info']['Total Cost (USD)']

    databases.insert(0, "Overview")
    enrichment_summaries_pivoted[0].insert(0, enrichment_overview)
    enrichment_summaries_pivoted[1].insert(0, "")
    enrichment_summaries_pivoted[2].insert(0, "")
    highlevel_df = pd.DataFrame(
        {
            "Enrichment Database": pd.Series(databases),
            "Results": pd.Series(enrichment_summaries_pivoted[0]),
            "Text": pd.Series(enrichment_summaries_pivoted[2]),
            "Source": pd.Series(enrichment_summaries_pivoted[1]),
        }
    )

    progress.update(overall_task, advance=15)
    
    # 7. build the report file
    progress.update(task_id=0, advance=5, description="Building final report...")
    model_match = re.search(r"model_name='([\w\-.]+)'", brad.state['process']['steps'][0]['llm'])
    temperature_match = re.search(r"temperature=([\d\.]+)", brad.state['process']['steps'][0]['llm'])
    
    model_name = model_match.group(1) if model_match else None
    temperature = float(temperature_match.group(1)) if temperature_match else None
    build_report_file(output_file, highlevel_df, enrichment_dfs, databases, gene_list=genes, query=query, model_name=model_name, temperature=temperature, cost=total_cost)
    
    # Finalize progress
    progress.update(overall_task, advance=5, description="Completed!")

"""
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
        help="Minimum number of enrichment terms to report (default: 3)."
    )

    parser.add_argument(
        "--maximum_enrichment_terms",
        type=int,
        default=10,
        help="Maximum number of enrichment terms to report (default: 10)."
    )

    parser.add_argument(
        "--literature_database",
        type=str,
        default="../databases/enrichment_database"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="enrichment_results.xlsx"
    )

    parser.add_argument(
        "--query",
        type=str,
        default=DEFAULT_QUERY
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose mode for debugging and logging."
    )

    # Parse arguments
    args = parser.parse_args()
    DATABASE_FOLDER = args.literature_database
    query = args.query if args.query != DEFAULT_QUERY else DEFAULT_QUERY

    if args.verbose:
        logger.info(f"Output file: {args.output}")
        logger.info(f"Literature database: {args.literature_database}")

    # Run the enrichment function
    results = perform_enrichment(
        gene_string=args.gene_string,
        databases=args.databases,
        threshold_p_value=args.threshold_p_value,
        minimum_enrichment_terms=args.minimum_enrichment_terms,
        maximum_enrichment_terms=args.maximum_enrichment_terms,
        literature_database=args.literature_database,
        query=query,
        output_file=args.output,
        verbose=args.verbose
    )
"""

@click.command()
@click.argument("gene_string", type=str)
@click.option(
    "--databases", "-d", multiple=True, 
    default=['KEGG_2021_Human', 'GO_Biological_Process_2021', 'PanglaoDB_Augmented_2021'], 
    help="List of databases to use for enrichment (default: KEGG, GO, PanglaoDB)."
)
@click.option(
    "--threshold-p-value", "-p", type=float, default=0.05, 
    help="P-value threshold for enrichment results (default: 0.05)."
)
@click.option(
    "--minimum-enrichment-terms", "-min", type=int, default=3, 
    help="Minimum number of enrichment terms to report (default: 3)."
)
@click.option(
    "--maximum-enrichment-terms", "-max", type=int, default=10, 
    help="Maximum number of enrichment terms to report (default: 10)."
)
@click.option(
    "--literature-database", "-l", type=str, default="default_db",
    help="Path to the literature database (default: ../databases/enrichment_database)."
)
@click.option(
    "--output", "-o", type=str, default="enrichment_results.xlsx",
    help="Output file name for results (default: enrichment_results.xlsx)."
)
@click.option(
    "--query", "-q", type=str, default=DEFAULT_QUERY, 
    help="Custom query for enrichment analysis (default: standard query)."
)
@click.option(
    "--verbose", "-v", is_flag=True, 
    help="Enable verbose mode for debugging and logging."
)
def cli(gene_string, databases, threshold_p_value, minimum_enrichment_terms, maximum_enrichment_terms, 
        literature_database, output, query, verbose):
    """
    Perform gene enrichment analysis using BRAD.
    
    GENE_STRING: Comma-separated list of genes (e.g., MYOD, P53, CDK2).
    """
    global overall_task
    progress.start()
    overall_task = progress.add_task("[green]Setting up...", total=100)

    if literature_database == "default_db":
        literature_database = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "databases", "enrichment_database"))
    
    if verbose:
        logger.info(f"Output file: {output}")
        logger.info(f"Literature database: {literature_database}")

    DATABASE_FOLDER = literature_database
    query = query if query != DEFAULT_QUERY else DEFAULT_QUERY

    # Run the enrichment function
    results = perform_enrichment(
        gene_string=gene_string,
        databases=list(databases),  # Convert tuple to list
        threshold_p_value=threshold_p_value,
        minimum_enrichment_terms=minimum_enrichment_terms,
        maximum_enrichment_terms=maximum_enrichment_terms,
        literature_database=literature_database,
        query=query,
        output_file=output,
        verbose=verbose
    )

    progress.stop()
    sys.stdout, sys.stderr = original_stdout, original_stderr
    if 'null_output' in locals():
        null_output.close()

if __name__ == "__main__":
    try:
        cli()
    finally:
        progress.stop()
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        if 'null_output' in locals():
            null_output.close()
    
