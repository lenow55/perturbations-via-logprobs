import asyncio
import csv
from functools import partial
import json
import logging
import os
from argparse import Namespace

import pandas as pd
from openai import AsyncOpenAI

from src.config import AppSettings, ChatLLMConfig, EmbedLLMConfig, MetadataFileInfo
from src.params import parser
from src.schemas import CheckLlmIn, CheckStage1Out, TA_logprob_list
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
    passages: dict[str, str],
    client: AsyncOpenAI,
    client_embed: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    config: AppSettings,
):
    if not isinstance(config.embed, EmbedLLMConfig):
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
    if not isinstance(args.out_folder, str):
        raise RuntimeError("Bad argument for report value")
    if not isinstance(args.config, str):
        raise RuntimeError("Bad argument for config path value")

    with open(args.config, "r") as f:
        config = AppSettings.model_validate_json(f.read())

    configure_logging(config=config)

    meta_file = os.path.join(args.out_folder, "meta.json")
    if not os.path.isdir(args.out_folder):
        logger.warning(f"Try create out_folder {args.out_folder}")
        os.mkdir(args.out_folder)
    else:
        if os.path.exists(meta_file):
            logger.error(f"Metadata file {meta_file} exists in target dir: ABORT!!!")
            return
        else:
            logger.info("Empty dir already exists")

    in_passages = os.path.join(args.input, "passages.json")
    in_checks = os.path.join(args.input, "checks.csv")
    with open(in_passages, "r") as f:
        passages = json.load(f)
    logger.info(f"Readed {in_passages} passages {len(passages)}")
    checks_df = pd.read_csv(in_checks, quoting=csv.QUOTE_NONNUMERIC, index_col=0)
    checks_df.index = checks_df.index.astype(int)
    logger.info(f"Checks readed from {in_checks} shape: {checks_df.shape}")

    client = create_openai_client(config=config.llm)
    if not isinstance(config.embed, EmbedLLMConfig):
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
                        "answer": str(row["answer"]),
                        "passage_id": str(int(row["passage_id"])),
                        "question": str(row["question"]),
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

    meta_info = MetadataFileInfo(**config.model_dump(mode="python"))
    with open(meta_file, "w") as f:
        _ = f.write(meta_info.model_dump_json(indent=2, exclude={"api_key"}))

    logger.info(f"Stage1 metadata saved into {meta_file} file")

    out_main = os.path.join(args.out_folder, "checks.csv")
    df = pd.DataFrame.from_records(
        data=results,
        index="check_id",
        exclude=["prompt_logprobs", "gen_answer"],
    )
    df.to_csv(out_main, quoting=csv.QUOTE_NONNUMERIC)
    logger.info(f"Stage1 results saved into {out_main} file; shape: {df.shape}")

    out_logprobs = os.path.join(args.out_folder, "gen_logprobs.csv")
    df_p = pd.DataFrame.from_records(
        data=results,
        index="check_id",
        exclude=["question", "answer", "passage_id", "similarity"],
    )
    ta_func = partial(TA_logprob_list.dump_python, mode="json")
    ta_func_json = partial(json.dumps, ensure_ascii=False)
    df_p["prompt_logprobs"] = df_p["prompt_logprobs"].map(ta_func).map(ta_func_json)
    df_p.to_csv(out_logprobs, quoting=csv.QUOTE_NONNUMERIC)

    logger.info(
        f"Stage1 prompt_logprobs saved into {out_logprobs} file; shape: {df.shape}"
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
