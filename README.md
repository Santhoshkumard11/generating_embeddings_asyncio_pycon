# Generate Embeddings - PyCon India 2023

This repo contains codes to generate embeddings using asyncio - 1k Azure openai calls in 5 minutes - PyCon India 2023

## Prerequisites

- Python 3.8 and above
- Install all packages from `requirements.txt`
- Set `AZURE_OPENAI_API_KEY` and `AZURE_OPENAI_ENDPOINT` environment variables

## How to run

- Create a CSV file called foo.csv with one column (text)
- Run `async_gen_embeddings.py`
- After a successful run it will generate a embeddings_foo.csv with a new column called embeddings
