import asyncio
import csv
from datetime import datetime
from functools import partial
import json
import logging
import os
from argparse import Namespace

import pandas as pd
from openai import AsyncOpenAI
from pydantic import BaseModel

from src.config import AppSettings, EmbedLLMConfig
from src.params import parser
from src.schemas import (
    Stage3In,
    Stage4Out,
    TA_logprob_list,
    TA_words_list,
)
from src.utils.base import (
    calculate_prompt_logprobs,
    calculate_similarity,
    check_out_folder_empty,
    configure_logging,
    create_openai_client,
)

logger = logging.getLogger(__name__)

# INFO: декодируем сразу весь запрос и ответ эталонный
# для получения логпробов на каждый токен


class MetadataFileInfo(BaseModel):
    date: datetime = datetime.now()
    input_folder: str
    output_folder: str
    config: AppSettings


async def stage_task(
    check: Stage3In,
    client: AsyncOpenAI,
    client_embed: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    config: AppSettings,
) -> Stage4Out:
    if not isinstance(config.embed, EmbedLLMConfig):
        logger.critical("Cant run without embed config")
        raise RuntimeError("bad config")

    text = "context: " + check["passage"] + "\nquestion: " + check["question"]
    answer, prompt_logprobs = await calculate_prompt_logprobs(
        idx=str(check["check_id"]),
        query=text,
        client=client,
        semaphore=semaphore,
        config=config.llm,
    )

    _, score = await calculate_similarity(
        idx=str(check["check_id"]),
        reference=check["answer"],
        answer=answer,
        client=client_embed,
        config=config.embed,
    )
    return Stage4Out(
        gen_answer=answer,
        prompt_logprobs=prompt_logprobs,
        similarity=score,
        **check,
    )


async def main(args: Namespace):
    if not isinstance(args.input, str):
        raise RuntimeError("Bad argument for input folder")
    if not isinstance(args.out_folder, str):
        raise RuntimeError("Bad argument for report value")
    if not isinstance(args.config, str):
        raise RuntimeError("Bad argument for config path value")

    with open(args.config, "r") as f:
        config = AppSettings.model_validate_json(f.read())

    configure_logging(config=config)
    meta_file = check_out_folder_empty(out_folder=args.out_folder)

    in_checks = os.path.join(args.input, "ptb_checks.csv")

    checks_df = pd.read_csv(in_checks, quoting=csv.QUOTE_NONNUMERIC, index_col=0)
    checks_df.index = checks_df.index.astype(int)
    logger.info(f"Checks PTB readed from {in_checks} shape: {checks_df.shape}")

    client = create_openai_client(config=config.llm)
    if not isinstance(config.embed, EmbedLLMConfig):
        logger.critical("Cant run without embed config")
        return

    client_embed = create_openai_client(config=config.embed)
    semaphore = asyncio.Semaphore(config.llm.async_cals)

    tasks: list[asyncio.Task[Stage4Out]] = []
    for idx, row in checks_df.iterrows():
        if not isinstance(idx, int):
            raise RuntimeError("Bad index type")
        tasks.append(
            asyncio.create_task(
                stage_task(
                    check={
                        "check_id": idx,
                        "answer": str(row["answer"]),
                        "passage_id": str(int(row["passage_id"])),
                        "question": str(row["question"]),
                        "passage": str(row["passage"]),
                        "ptb_words": str(row["ptb_words"]),
                    },
                    client=client,
                    client_embed=client_embed,
                    semaphore=semaphore,
                    config=config,
                )
            )
        )
    results = await asyncio.gather(*tasks)

    meta_info = MetadataFileInfo(
        input_folder=args.input,
        output_folder=args.out_folder,
        config=config,
    )
    with open(meta_file, "w") as f:
        _ = f.write(meta_info.model_dump_json(indent=2, exclude={"api_key"}))

    logger.info(f"Stage4 metadata saved into {meta_file} file")

    out_main = os.path.join(args.out_folder, "ptb_results.csv")
    df = pd.DataFrame.from_records(
        data=results,
        index="check_id",
        exclude=["prompt_logprobs", "gen_answer"],
    )
    ta_func = partial(TA_words_list.dump_python, mode="json")
    ta_func_json = partial(json.dumps, ensure_ascii=False)
    df["ptb_words"] = df["ptb_words"].map(ta_func).map(ta_func_json)
    df.to_csv(out_main, quoting=csv.QUOTE_NONNUMERIC)
    logger.info(f"Stage4 prompt_logprobs saved into {out_main} file; shape: {df.shape}")

    out_logprobs = os.path.join(args.out_folder, "gen_logprobs.csv")
    df_p = pd.DataFrame.from_records(
        data=results,
        index="check_id",
        exclude=[
            "question",
            "answer",
            "passage_id",
            "similarity",
            "passage",
            "ptb_words",
        ],
    )
    ta_func = partial(TA_logprob_list.dump_python, mode="json")
    ta_func_json = partial(json.dumps, ensure_ascii=False)
    df_p["prompt_logprobs"] = df_p["prompt_logprobs"].map(ta_func).map(ta_func_json)
    df_p.to_csv(out_logprobs, quoting=csv.QUOTE_NONNUMERIC)

    logger.info(
        f"Stage4 prompt_logprobs saved into {out_logprobs} file; shape: {df.shape}"
    )


if __name__ == "__main__":
    _ = parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Путь до директории с датасетом",
    )
    _ = parser.add_argument(
        "-o",
        "--out-folder",
        type=str,
        required=True,
        help="Путь до директории с пробами и ответами",
    )
    asyncio.run(main(parser.parse_args()))
