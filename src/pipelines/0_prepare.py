import csv
import json
import logging
import os
from argparse import Namespace

import pandas as pd
from numpy.random import RandomState
from pydantic import TypeAdapter

from src.config import AppSettings
from src.params import parser
from src.schemas import Check, ReadingComprehensionItem
from src.utils.base import configure_logging

logger = logging.getLogger(__name__)

# INFO: декодируем сразу весь запрос и ответ эталонный
# для получения логпробов на каждый токен


def main(args: Namespace):
    if not isinstance(args.input, str):
        raise RuntimeError("Bad argument for input file")
    if not isinstance(args.seed, int):
        raise RuntimeError("Bad argument for seed value")
    if not isinstance(args.samples, int):
        raise RuntimeError("Bad argument for samples value")
    if not isinstance(args.out_folder, str):
        raise RuntimeError("Bad argument for out_folder value")
    if not isinstance(args.config, str):
        raise RuntimeError("Bad argument for config path value")
    if not isinstance(args.verdict, int):
        raise RuntimeError("Bad argument for verdict value")

    verdict = args.verdict

    with open(args.config, "r") as f:
        config = AppSettings.model_validate_json(f.read())

    configure_logging(config=config)
    logger.info(f"Verdict type: {verdict}")

    df = pd.read_json(args.input, lines=True)
    small_df = df.sample(n=args.samples, random_state=RandomState(seed=args.seed))
    del df

    small_list = small_df.to_dict(orient="records")
    ta = TypeAdapter(list[ReadingComprehensionItem])
    items = ta.validate_python(small_list)

    id2passage: dict[str, str] = {}
    id2check: dict[int, Check] = {}

    logger.info(f"Read {len(items)} from source file")
    for item in items:
        id2passage.update({str(item.idx): item.passage.text.strip(' "')})
        for question in item.passage.questions:
            for answer in question.answers:
                if answer.label == verdict:
                    id2check.update(
                        {
                            answer.idx: {
                                "question": question.question.strip(' "'),
                                "answer": answer.text.strip(' "'),
                                "passage_id": str(item.idx),
                            }
                        }
                    )
    logger.info(f"Generated {len(id2check)} checks")

    if not os.path.isdir(args.out_folder):
        logger.warning(f"Try create out_folder {args.out_folder}")
        os.mkdir(args.out_folder)

    out_passages = os.path.join(args.out_folder, "passages.json")
    with open(out_passages, "w") as f:
        json.dump(id2passage, f, indent=2, ensure_ascii=False)
    logger.info(f"Passages saved into {out_passages} file")

    check_df = pd.DataFrame.from_dict(id2check, orient="index")
    logger.info(f"Convert checks into dataframe: {check_df.shape}")

    out_checks = os.path.join(args.out_folder, "checks.csv")
    check_df.to_csv(out_checks, quoting=csv.QUOTE_NONNUMERIC)
    logger.info(f"Checks saved into {out_checks} file")


if __name__ == "__main__":
    _ = parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Путь до файла jsonl с MuSeRC",
    )
    _ = parser.add_argument(
        "-o",
        "--out-folder",
        type=str,
        required=True,
        help="Путь до директории с подготовленными датасетами",
    )
    _ = parser.add_argument(
        "-n",
        "--samples",
        type=int,
        required=False,
        help="Количество элементов в подвыборке",
        default=10,
    )
    _ = parser.add_argument(
        "-s",
        "--seed",
        type=int,
        required=False,
        help="Seed для выборки элементов",
        default=42,
    )
    _ = parser.add_argument(
        "-v",
        "--verdict",
        type=int,
        required=False,
        help="1 - правильный ответ, 0 - неправильный ответ",
        default=1,
    )
    main(parser.parse_args())
