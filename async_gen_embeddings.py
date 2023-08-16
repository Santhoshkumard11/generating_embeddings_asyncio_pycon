import os
import asyncio

# import aiofiles
from time import time
import logging

import pandas as pd
import tiktoken
import openai
from openai.embeddings_utils import aget_embedding
from tenacity import (
    retry,
    stop_after_attempt,
    retry_if_exception_type,
    wait_fixed,
    after_log,
)
from openai.error import RateLimitError, APIError, Timeout

logger = logging.getLogger("generate_embeddings")

API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
RESOURCE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")

openai.api_type = "azure"
openai.api_key = API_KEY
openai.api_base = RESOURCE_ENDPOINT
openai.api_version = "2022-12-01"

url = openai.api_base + "/openai/deployments?api-version=2022-12-01"

embedding_engine = "text-embedding-ada-002"
tokenizer = tiktoken.get_encoding("cl100k_base")


input_csv_file_name = "foo.csv"
output_csv_file_name = "embeddings_foo.csv"


def calculate_tokens(df):
    """Return only rows with less than 8k tokens as a pandas dataframe"""
    df["n_tokens"] = df["text"].apply(lambda x: len(tokenizer.encode(x)))

    # filter out texts which are more than 8k tokens
    df = df.query("n_tokens < 8192")


# wait for 120 seconds before making another call
# try six time before throwing an exception
@retry(
    wait=wait_fixed(120),
    stop=stop_after_attempt(6),
    retry=retry_if_exception_type((RateLimitError, APIError, Timeout)),
    after=after_log(logger, logging.INFO),
)
async def get_embeddings(index, input_paragraph):
    """Make calls to embeddings generation API and return the result

    Args:
        index (int): index of the input paragraph from pandas dataframe
        input_paragraph (str): input text to create embeddings

    Returns:
        tuple(int,List[float]): list of index and embeddings
    """
    logger.info(f"Processing index - {index}")

    text_embeddings = await aget_embedding(
        input=input_paragraph, engine=embedding_engine
    )

    logger.info(f"Done index - {index}")

    # use it only if we've to store the results - in case of a failure
    # async with aiofiles.open("temp_embeddings.txt") as f:
    #     await f.write(f"{index},{text_embeddings}\n")
    #     await f.flush()

    return index, text_embeddings


async def generate_embeddings():
    """Return generated embeddings for the input csv file and output a new csv file with embeddings"""
    input_df = pd.read_csv(input_csv_file_name)

    # schedule all the coroutines and wait for them to complete
    list_gen_text_embeddings = await asyncio.gather(
        *[
            generate_embeddings(index, data["text"])
            for index, data in input_df.iterrows()
        ]
    )

    # sort based on the first index - which is the dataframe index
    # because we don't know which order the embeddings have completed
    sorted_list_gen_text_embeddings = sorted(
        list_gen_text_embeddings, key=lambda x: x[0]
    )

    # add embeddings to the dataframe
    input_df["embeddings"] = sorted_list_gen_text_embeddings

    input_df.to_csv(output_csv_file_name)


async def main():
    start_time = time()
    logger.info("Starting to generate embeddings")
    await generate_embeddings()
    end_time = time()
    logger.info(f"Done generating embeddings in {round(end_time - start_time,2)}s")


if __name__ == "__main__":
    asyncio.run(main())
