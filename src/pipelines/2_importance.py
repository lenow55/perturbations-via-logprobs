import asyncio
import csv
from functools import partial
import json
import logging
import os
from argparse import Namespace
from datetime import datetime
from typing import Any

import pandas as pd
from openai import AsyncOpenAI
from pydantic import BaseModel, SerializeAsAny

from src.config import AppSettings
from src.params import parser
from src.schemas import (
    CheckLlmIn,
    Stage2Out,
    TA_logprob_list,
    TA_tokens_list,
    TA_words_list,
)
from src.utils.base import (
    configure_logging,
)
from src.utils.word_analyzer import WordAnalyzerBuilder

logger = logging.getLogger(__name__)


class MetadataFileInfo(BaseModel):
    date: datetime = datetime.now()
    input_folder: str
    stage_folder: str
    output_folder: str
    metadata: SerializeAsAny[dict[Any, Any]]


def stage_task(
    check: CheckLlmIn,
    passages: dict[int, str],
    client: AsyncOpenAI,
    client_embed: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    config: AppSettings,
):
    pass


def main(args: Namespace):
    if not isinstance(args.input, str):
        raise RuntimeError("Bad argument for input folder")
    if not isinstance(args.stage, str):
        raise RuntimeError("Bad argument for stage file")
    if not isinstance(args.output, str):
        raise RuntimeError("Bad argument for report value")
    if not isinstance(args.config, str):
        raise RuntimeError("Bad argument for config path value")
    if not isinstance(args.token_metric, str):
        raise RuntimeError("Bad token-metric value")
    if not isinstance(args.word_metric, str):
        raise RuntimeError("Bad word-metric value")

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

    in_checks = os.path.join(args.stage, "checks.csv")
    in_logprobs = os.path.join(args.stage, "gen_logprobs.csv")
    in_passages = os.path.join(args.input, "passages.json")

    # файлик с контекстами
    with open(in_passages, "r") as f:
        passages = json.load(f)
    logger.info(f"Readed {in_passages} passages {len(passages)}")

    # датасет с колонками
    # ["question", "answer", "passage_id", "similarity"]
    # индекс "check_id"
    checks_df = pd.read_csv(in_checks, quoting=csv.QUOTE_NONNUMERIC, index_col=0)
    checks_df.index = checks_df.index.astype(int)
    checks_df["passage_id"] = checks_df["passage_id"].astype(int).astype(str)
    logger.info(f"Checks readed from {in_checks} shape: {checks_df.shape}")

    # датасет с колонками
    # ["prompt_logprobs", "gen_answer"]
    # индекс "check_id"
    logprobs_df = pd.read_csv(in_logprobs, quoting=csv.QUOTE_NONNUMERIC, index_col=0)
    logprobs_df.index = logprobs_df.index.astype(int)
    logger.info(
        f"Logprobs dataset readed from {in_logprobs} shape: {logprobs_df.shape}"
    )

    builder = WordAnalyzerBuilder()
    builder.set_token_metric(args.token_metric)
    builder.set_word_metric(args.word_metric)
    analyzer = builder.build()

    stage2res_list: list[Stage2Out] = []
    for check_id, row in checks_df.iterrows():
        if not isinstance(check_id, int):
            continue
        logprobs_j = logprobs_df.loc[check_id, "prompt_logprobs"]
        if not isinstance(logprobs_j, str):
            continue
        logprobs = TA_logprob_list.validate_json(logprobs_j)

        tokens_importances = analyzer.tokens_importances(prompt_logprobs=logprobs)
        passage_id = checks_df.loc[check_id, "passage_id"]
        if not isinstance(passage_id, str):
            continue
        passage = passages[passage_id]
        words_importances = analyzer.words_importances(
            tokens_importances=tokens_importances, passage=passage
        )
        stage2res_list.append(
            Stage2Out(
                check_id=check_id,
                words_importances=words_importances,
                tokens_importances=tokens_importances,
            )
        )

    # INFO: сохранение

    out_main = os.path.join(args.out_folder, "importances.csv")
    meta_info = MetadataFileInfo(
        input_folder=args.input,
        stage_folder=args.stage,
        output_folder=args.output,
        metadata=builder.metadata,
    )
    with open(meta_file, "w") as f:
        _ = f.write(meta_info.model_dump_json(indent=2))

    df = pd.DataFrame.from_records(data=stage2res_list, index="check_id")

    ta_func = partial(TA_words_list.dump_python, mode="json")
    ta_func_json = partial(json.dumps, ensure_ascii=False)
    df["words_importances"] = df["words_importances"].map(ta_func).map(ta_func_json)

    ta_func = partial(TA_tokens_list.dump_python, mode="json")
    ta_func_json = partial(json.dumps, ensure_ascii=False)
    df["tokens_importances"] = df["tokens_importances"].map(ta_func).map(ta_func_json)

    df.to_csv(out_main, quoting=csv.QUOTE_NONNUMERIC)
    logger.info(f"Stage2 results saved into {out_main} file; shape: {df.shape}")


if __name__ == "__main__":
    _ = parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Путь до директории с датасетом",
    )
    _ = parser.add_argument(
        "-s",
        "--stage",
        type=str,
        required=True,
        help="Путь файла с стадии 1",
    )
    _ = parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Путь до файла с результатом подсчёта значимости",
    )
    _ = parser.add_argument(
        "-t",
        "--token-metric",
        type=str,
        required=True,
        help="Код функции подсчёта метрики для токена",
    )
    _ = parser.add_argument(
        "-w",
        "--word-metric",
        type=str,
        required=True,
        help="Код функции подсчёта метрики для слова",
    )
    main(parser.parse_args())
