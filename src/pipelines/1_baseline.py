import asyncio
import csv
import json
import logging
import os
from argparse import Namespace

import pandas as pd
from openai import AsyncOpenAI

from src.config import AppSettings, ChatLLMConfig, EmbedLLMConfig
from src.params import parser
from src.schemas import CheckLlmIn, CheckStage1Out
from src.utils.base import (
    calculate_prompt_logprobs,
    calculate_similarity,
    configure_logging,
    create_openai_client,
)

logger = logging.getLogger(__name__)

# INFO: декодируем сразу весь запрос и ответ эталонный
# для получения логпробов на каждый токен


async def stage_task(
    check: CheckLlmIn,
    passages: dict[int, str],
    client: AsyncOpenAI,
    client_embed: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    config: AppSettings,
):
    if not isinstance(config.embed, ChatLLMConfig):
        logger.critical("Cant run without embed config")
        raise RuntimeError("bad config")

    text = (
        "context: " + passages[check["passage_id"]] + "\nquestion: " + check["question"]
    )
    answer, prompt_logprobs = await calculate_prompt_logprobs(
        idx=str(check["check_id"]),
        query=text,
        client=client,
        semaphore=semaphore,
        config=config.llm,
    )

    if not isinstance(config.embed, EmbedLLMConfig):
        raise ValueError("Configuration for embeding model not found")

    _, score = await calculate_similarity(
        idx=str(check["check_id"]),
        reference=check["answer"],
        answer=answer,
        client=client_embed,
        config=config.embed,
    )
    return CheckStage1Out(
        gen_answer=answer,
        prompt_logprobs=prompt_logprobs,
        similarity=score,
        **check,
    )


async def main(args: Namespace):
    if not isinstance(args.input, str):
        raise RuntimeError("Bad argument for input folder")
    if not isinstance(args.output, str):
        raise RuntimeError("Bad argument for report value")
    if not isinstance(args.config, str):
        raise RuntimeError("Bad argument for config path value")

    with open(args.config, "r") as f:
        config = AppSettings.model_validate_json(f.read())

    configure_logging(config=config)

    in_passages = os.path.join(args.input, "passages.json")
    in_checks = os.path.join(args.input, "checks.csv")
    with open(in_passages, "r") as f:
        passages = json.load(f)
    logger.info(f"Readed {in_passages} passages {len(passages)}")
    checks_df = pd.read_csv(in_checks, quoting=csv.QUOTE_NONNUMERIC)
    logger.info(f"Checks readed from {in_checks} shape: {checks_df.shape}")

    client = create_openai_client(config=config.llm)
    if not isinstance(config.embed, ChatLLMConfig):
        logger.critical("Cant run without embed config")
        return

    client_embed = create_openai_client(config=config.embed)
    semaphore = asyncio.Semaphore(config.llm.async_cals)

    tasks: list[asyncio.Task[CheckStage1Out]] = []
    for idx, row in checks_df.iterrows():
        if not isinstance(idx, int):
            raise RuntimeError("Bad index type")
        tasks.append(
            asyncio.create_task(
                stage_task(
                    check={
                        "check_id": idx,
                        "answer": row["answer"],
                        "passage_id": row["passage_id"],
                        "question": row["question"],
                    },
                    passages=passages,
                    client=client,
                    client_embed=client_embed,
                    semaphore=semaphore,
                    config=config,
                )
            )
        )
    results = await asyncio.gather(*tasks)

    df = pd.DataFrame.from_records(data=results, index="check_id")
    df.to_csv(args.output, quoting=csv.QUOTE_NONNUMERIC)
    logger.info(f"Stage1 results saved into {args.output} file; shape: {df.shape}")


if __name__ == "__main__":
    _ = parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Путь до директории с датасетом",
    )
    asyncio.run(main(parser.parse_args()))
