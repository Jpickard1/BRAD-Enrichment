# Gene Enrichment with BRAD

BRAD Enrichment is a command-line tool to generate a report for gene enrichment analysis based on a gene set and targeted literature database. The tool uses a [BRAD Agent](https://github.com/Jpickard1/BRAD) to identify the contextual significance of enrichment terms and databases based upon custom a custom literature database. This repository contains command line tools `brad-builddb` and `brad-enrichment` for building literature databases and generating reports.

![image](https://github.com/user-attachments/assets/bcab41bf-d57e-4aec-ad95-0aab14a81969)

## Quickstart
Run the following commands to install this tool:
```
pip install brad-enrichment
brad-builddb --help
brad-enrichment --help
```
See below for detailed list on installing this code.

## Installation
To install BRAD Enrichment, activate your conda environment from the [BRAD repository](https://github.com/Jpickard1/BRAD) and install the following two packages:
```sh
conda activate BRAD
pip install BRAD-Agent
pip install brad-enrichment
```

## Usage
### Gene Enrichment Analysis
To perform gene enrichment analysis, use the following command:

```sh
brad-enrichment <gene_string> [OPTIONS]
```

#### Arguments
- `<gene_string>`: A string containing gene names separated by spaces or commas.

#### Options
- `--databases`, `-d`: List of databases to use for enrichment (default: KEGG, GO, PanglaoDB).
- `--threshold-p-value`, `-p`: P-value threshold for enrichment results (default: 0.05).
- `--minimum-enrichment-terms`, `-min`: Minimum number of enrichment terms to report (default: 3).
- `--maximum-enrichment-terms`, `-max`: Maximum number of enrichment terms to report (default: 10).
- `--literature-database`, `-l`: Path to the literature database (default: `../databases/enrichment_database`).
- `--output`, `-o`: Output file name for results (default: `enrichment_results.xlsx`).
- `--query`, `-q`: Custom query for enrichment analysis (default: standard query).
- `--verbose`, `-v`: Enable verbose mode for debugging and logging.

#### Example Usage
```sh
brad-enrichment "TP53, MYC, EGFR" -d KEGG_2021_Human -p 0.01 -o my_results.xlsx
```

### Building an Enrichment Literature Database
To build an enrichment literature database, use the following command:

```sh
brad-builddb [OPTIONS]
```

#### Options
- `--documents-directory`, `-d`: Path to the directory containing document files (default: `documents`).
- `--database-directory`, `-D`: Path to the directory where the database should be stored (default: `databases`).
- `--database-name`, `-n`: Name of the database to be created (default: `enrichment_database`).
- `--text-size`, `-s`: Size of text chunks for processing (default: 700).
- `--text-overlap`, `-o`: Number of overlapping characters between chunks (default: 100).
- `--verbose`, `-v`: Enable verbose output.

#### Example Usage
```sh
brad-builddb -d /path/to/documents -D /path/to/databases -n my_database -s 500 -o 50 -v
```

## Development Setup
If you wish to contribute or modify the tool, follow these steps:

```sh
git clone https://github.com/Jpickard1/BRAD-Enrichment.git
cd BRAD-Enrichment
pip install -e .
```

## Citation
If you use this tool in your research, please cite or paper [Language Model Powered Digital Biology with BRAD](https://arxiv.org/abs/2409.02864) as:

```
@article{pickard2024language,
  title={Language Model Powered Digital Biology with BRAD},
  author={Pickard, Joshua and Prakash, Ram and Choi, Marc Andrew and Oliven, Natalie and
          Stansbury, Cooper and Cwycyshyn, Jillian
          and Gorodetsky, Alex and Velasquez, Alvaro and Rajapakse, Indika},
  journal={arXiv preprint arXiv:2409.02864},
  url={https://arxiv.org/abs/2409.02864},
  year={2024}
}
```
