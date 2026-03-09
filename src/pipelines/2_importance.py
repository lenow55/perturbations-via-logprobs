import asyncio
import csv
import json
import logging
import os
from argparse import Namespace
from datetime import datetime

import pandas as pd
from openai import AsyncOpenAI
from pydantic import BaseModel

from src.config import AppSettings
from src.params import parser
from src.schemas import CheckLlmIn
from src.utils.base import (
    configure_logging,
)
from src.utils.metrics_hub import HUB

logger = logging.getLogger(__name__)


class MetadataFileInfo(BaseModel):
    date: datetime = datetime.now()
    input_folder: str
    stage_folder: str
    output_folder: str
    metric_name: str


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
    if not isinstance(args.metric, str):
        raise RuntimeError("Bad metric value")
    if args.metric not in HUB:
        raise RuntimeError(f"Metric '{args.metric}' not in HUB")

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
    logger.info(f"Checks readed from {in_checks} shape: {checks_df.shape}")

    # датасет с колонками
    # ["prompt_logprobs", "gen_answer"]
    # индекс "check_id"
    logprobs_df = pd.read_csv(in_logprobs, quoting=csv.QUOTE_NONNUMERIC, index_col=0)
    logprobs_df.index = logprobs_df.index.astype(int)
    logger.info(
        f"Logprobs dataset readed from {in_logprobs} shape: {logprobs_df.shape}"
    )

    # INFO: сохранение

    meta_info = MetadataFileInfo(
        input_folder=args.input,
        stage_folder=args.stage,
        output_folder=args.output,
        metric_name=args.metric,
    )
    with open(meta_file, "w") as f:
        _ = f.write(meta_info.model_dump_json(indent=2))

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
        "-m",
        "--metric",
        type=str,
        required=True,
        help="Код функции подсчёта метрики",
    )
    main(parser.parse_args())
