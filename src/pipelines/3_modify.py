import asyncio
import csv
import json
import logging
import os
import traceback
from argparse import Namespace
from datetime import datetime
from functools import partial
from operator import itemgetter
from types import UnionType
from typing import Any

import pandas as pd
from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict, Field, create_model

from src.config import AppSettings
from src.params import parser
from src.schemas import (
    CheckLlmIn,
    Stage3Out,
    TA_words_list,
    WordImportance,
)
from src.utils.base import (
    check_out_folder_empty,
    configure_logging,
    create_openai_client,
)

logger = logging.getLogger(__name__)


class LLMMismatch(Exception):
    pass


def modify_text_and_get_indices(text: str, replacements: list[WordImportance]):
    # Сортируем список замен по начальному индексу (слева направо)
    # Это обязательно, чтобы смещение работало корректно
    replacements.sort(key=itemgetter("start"))

    modified_text = text
    new_positions: list[WordImportance] = []  # Здесь будем хранить новые координаты
    offset: int = 0  # Текущее смещение индексов

    for replacement in replacements:
        # 1. Вычисляем точную позицию начала в текущей (уже частично измененной) строке
        actual_start = replacement["start"] + offset

        # 2. Вычисляем новый конец (это просто начало + длина нового слова)
        actual_end = actual_start + len(replacement["word"])

        new_word = replacement["word"]

        # 3. Сохраняем информацию о новом положении слова
        new_positions.append(
            {
                "word": new_word,
                "start": actual_start,
                "end": actual_end,
                "importance": replacement["importance"],
            }
        )

        # 4. Производим замену в строке
        # modified_text[end + offset:] берет остаток строки после старого слова
        modified_text = (
            modified_text[:actual_start]
            + new_word
            + modified_text[replacement["end"] + offset :]
        )

        # 5. Пересчитываем смещение для следующих слов
        # Разница между длиной вставленного и вырезанного фрагментов
        offset += len(new_word) - (replacement["end"] - replacement["start"])

    return modified_text, new_positions


def schema_generator(words: list[WordImportance]) -> type[BaseModel]:
    fields: dict[str, tuple[type | UnionType, Any]] = {}
    for i, word in enumerate(words):
        name = f"{i}:" + word["word"]
        field: dict[str, tuple[type | UnionType, Any]] = {
            name: (str, Field(alias=name))
        }
        fields.update(field)

    config = ConfigDict(title="Слова")
    temp_model = create_model(
        "Слова",
        __config__=config,
        __doc__=None,
        __base__=None,
        __module__=__name__,
        __validators__=None,
        __cls_kwargs__=None,
        **fields,
    )
    return temp_model


async def stage_task(
    check: CheckLlmIn,
    words_importances: list[WordImportance],
    passages: dict[str, str],
    count_perturb: int,
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    config: AppSettings,
):

    llm_conf = config.llm

    count_top = min(count_perturb, len(words_importances))
    sorted_words = sorted(words_importances, key=itemgetter("importance"), reverse=True)
    top_words = sorted_words[:count_top]
    out_schema = schema_generator(words=top_words)

    async with semaphore:
        logger.debug(f"Start request id {check['check_id']}, use {count_top} words")

        response = await client.chat.completions.parse(
            model=llm_conf.model,
            messages=[
                {
                    "role": "system",
                    "content": "Ты лингвист русского языка и исполняешь задания.",
                },
                {
                    "role": "user",
                    "content": f"""
Задание:
Тебе нужно придумать противоположные слова или слова замены для ключей в переданой схеме

Действия:
1. считай слово без профикса вида i: (1: 2: и тд)
2. придумай другое слово вместо предоставленного.
например: он - она, оратор - слушатель, взял - положил, пришёл - ушёл
3. Замени имена и названия на придуманные:
например: Илья - Аня, Ланит - ChatGPT, Волга - Луна

Выходная схема:
{json.dumps(out_schema.model_json_schema(), indent=1)}

Задание:
Тебе нужно придумать противоположные слова или слова замены для ключей в переданой схеме

Действия:
1. считай слово без профикса вида i: (1: 2: и тд)
2. придумай другое слово вместо предоставленного.
например: он - она, оратор - слушатель, взял - положил, пришёл - ушёл
3. Замени имена и названия на придуманные:
например: Илья - Аня, Ланит - ChatGPT, Волга - Луна
""",
                },
            ],
            extra_body=llm_conf.extra_body,
            response_format=out_schema,
            **llm_conf.params_extra,
        )

    ptb_words_m = response.choices[0].message.parsed
    if not isinstance(ptb_words_m, BaseModel):
        raise RuntimeError("Error when llm generate")
    ptb_words_d = ptb_words_m.model_dump(mode="python", by_alias=True)
    logger.debug(ptb_words_m.model_dump_json(indent=2))

    ptb_words: list[WordImportance] = []
    for i, word in enumerate(top_words):
        key = f"{i}:" + word["word"]
        value = ptb_words_d.get(key, None)
        if not isinstance(value, str):
            continue
        ptb_words.append(
            WordImportance(
                word=value,
                start=word["start"],
                end=word["end"],
                importance=-1.0,
            )
        )
    if len(top_words) and not len(ptb_words):
        logger.error(f"Request id {check['check_id']}: llm return empty list")
        logger.debug(
            json.dumps(
                TA_words_list.dump_python(ptb_words), indent=2, ensure_ascii=False
            )
        )
        raise LLMMismatch

    # INFO: дальше надо новые слова внедрить в контекст

    try:
        passage = passages[check["passage_id"]]
    except KeyError:
        logger.error(
            f"Request id {check['check_id']}: passage {check['passage_id']} not found"
        )
        raise KeyError

    passage, ptb_words = modify_text_and_get_indices(
        text=passage, replacements=ptb_words
    )

    return Stage3Out(
        answer=check["answer"],
        question=check["question"],
        passage_id=check["passage_id"],
        check_id=check["check_id"],
        passage=passage,
        ptb_words=ptb_words,
    )


class MetadataFileInfo(BaseModel):
    date: datetime = datetime.now()
    input_folder: str
    stage_folder: str
    output_folder: str
    count_perturb: int
    config: AppSettings


async def main(args: Namespace):
    if not isinstance(args.input, str):
        raise RuntimeError("Bad argument for input folder")
    if not isinstance(args.stage, str):
        raise RuntimeError("Bad argument for stage file")
    if not isinstance(args.out_folder, str):
        raise RuntimeError("Bad argument for report value")
    if not isinstance(args.count_perturb, int):
        raise RuntimeError("Bad argument for count_perturb value")
    if not isinstance(args.config, str):
        raise RuntimeError("Bad argument for config path value")

    with open(args.config, "r") as f:
        config = AppSettings.model_validate_json(f.read())

    configure_logging(config=config)
    meta_file = check_out_folder_empty(out_folder=args.out_folder)

    in_importances = os.path.join(args.stage, "importances.csv")
    in_checks = os.path.join(args.input, "checks.csv")
    in_passages = os.path.join(args.input, "passages.json")

    # файлик с контекстами
    with open(in_passages, "r") as f:
        passages = json.load(f)
    logger.info(f"Readed {in_passages} passages {len(passages)}")

    checks_df = pd.read_csv(in_checks, quoting=csv.QUOTE_NONNUMERIC, index_col=0)
    checks_df.index = checks_df.index.astype(int)
    checks_df["passage_id"] = checks_df["passage_id"].astype(int).astype(str)
    logger.info(f"Checks readed from {in_checks} shape: {checks_df.shape}")

    importances_df = pd.read_csv(
        in_importances, quoting=csv.QUOTE_NONNUMERIC, index_col=0
    )
    importances_df.index = importances_df.index.astype(int)
    logger.info(
        f"Importances readed from {in_importances} shape: {importances_df.shape}"
    )

    client = create_openai_client(config=config.llm)
    semaphore = asyncio.Semaphore(config.llm.async_cals)

    tasks: list[asyncio.Task[Stage3Out]] = []
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
                    words_importances=TA_words_list.validate_json(
                        str(importances_df.loc[idx, "words_importances"])
                    ),
                    count_perturb=args.count_perturb,
                    passages=passages,
                    client=client,
                    semaphore=semaphore,
                    config=config,
                )
            )
        )
    results: list[Stage3Out] = []
    counter = 0
    try:
        for task in asyncio.as_completed(tasks):
            try:
                cluster_prop = await task
                results.append(cluster_prop)
            except LLMMismatch as e:
                continue
            except Exception as e:
                logger.warning(f"Error when generate ptbs, record will skip: {e}")
                logger.debug(traceback.format_exc())
                continue
            finally:
                counter = counter + 1
                if counter % 100 == 0:
                    logger.info(f"Processed {counter}/{len(tasks)}")
    except asyncio.exceptions.CancelledError:
        logger.error("Получен Ctrl+C, отменяем задачи...")
        for t in tasks:
            _ = t.cancel()
        _ = await asyncio.gather(*tasks, return_exceptions=True)
        logger.error("Все задачи корректно завершены")
        exit(1)

    meta_info = MetadataFileInfo(
        input_folder=args.input,
        stage_folder=args.stage,
        output_folder=args.out_folder,
        count_perturb=args.count_perturb,
        config=config,
    )
    with open(meta_file, "w") as f:
        _ = f.write(meta_info.model_dump_json(indent=2, exclude={"api_key"}))

    logger.info(f"Stage3 metadata saved into {meta_file} file")

    out_passages = os.path.join(args.out_folder, "ptb_checks.csv")
    df_p = pd.DataFrame.from_records(data=results, index="check_id")
    ta_func = partial(TA_words_list.dump_python, mode="json")
    ta_func_json = partial(json.dumps, ensure_ascii=False)
    df_p["ptb_words"] = df_p["ptb_words"].map(ta_func).map(ta_func_json)
    df_p.to_csv(out_passages, quoting=csv.QUOTE_NONNUMERIC)

    logger.info(
        f"Stage3 passage modifications saved into {out_passages} file; shape: {df_p.shape}"
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
        "-s",
        "--stage",
        type=str,
        required=True,
        help="Путь директории с стадии 2",
    )
    _ = parser.add_argument(
        "-o",
        "--out-folder",
        type=str,
        required=True,
        help="Путь до директории с пробами и ответами",
    )
    _ = parser.add_argument(
        "-n",
        "--count-perturb",
        type=int,
        default=5,
        required=False,
        help="Количество модификаций контекста",
    )
    asyncio.run(main(parser.parse_args()))
