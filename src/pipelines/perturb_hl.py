import asyncio
import logging
import random
from argparse import Namespace

import pandas as pd
from numpy.random import RandomState
from pydantic import TypeAdapter

from src.config import AppSettings
from src.params import parser
from src.ptb_visual import generate_full_report
from src.schemas import PtbScenarioRes, ReadingComprehensionItem
from src.utils.base import configure_logging, create_openai_client
from src.utils.words import find_ptb_words

logger = logging.getLogger(__name__)

# INFO: декодируем сразу весь запрос и ответ эталонный
# для получения логпробов на каждый токен


async def main(args: Namespace):
    if not isinstance(args.input, str):
        raise RuntimeError("Bad argument for input file")
    if not isinstance(args.seed, int):
        raise RuntimeError("Bad argument for seed value")
    if not isinstance(args.samples, int):
        raise RuntimeError("Bad argument for samples value")
    if not isinstance(args.report, str):
        raise RuntimeError("Bad argument for report value")
    if not isinstance(args.model, str):
        raise RuntimeError("Bad argument for model value")
    if not isinstance(args.config, str):
        raise RuntimeError("Bad argument for config path value")

    with open(args.config, "r") as f:
        config = AppSettings.model_validate_json(f.read())

    configure_logging(config=config)

    df = pd.read_json(args.input, lines=True)
    small_df = df.sample(n=args.samples, random_state=RandomState(seed=args.seed))
    del df

    small_list = small_df.to_dict(orient="records")
    ta = TypeAdapter(list[ReadingComprehensionItem])
    items = ta.validate_python(small_list)

    client = create_openai_client(config=config.llm)
    semaphore = asyncio.Semaphore(config.llm.async_cals)

    random.seed(a=args.seed)
    tasks: list[asyncio.Task[tuple[str, PtbScenarioRes]]] = []
    for i, item in enumerate(items):
        context = item.passage.text
        question = random.choice(item.passage.questions)
        answer = random.choice(question.answers)

        tasks.append(
            asyncio.create_task(
                find_ptb_words(
                    idx=str(item.idx),
                    scenario={
                        "name": f"Запрос {i}",
                        "context": context,
                        "reference": answer.text,
                        "question": question.question,
                    },
                    client=client,
                    semaphore=semaphore,
                    model=args.model,
                    config=config.llm,
                )
            )
        )
    results = await asyncio.gather(*tasks)

    generate_full_report(scenario_results=results, filename=args.report, config=config)


if __name__ == "__main__":
    _ = parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Путь до файла jsonl с MuSeRC",
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
        "-w",
        "--words",
        type=int,
        required=False,
        help="Количество слов, которые ищем в контексте",
        default=5,
    )
    _ = parser.add_argument(
        "-s",
        "--seed",
        type=int,
        required=False,
        help="Seed для выборки элементов",
        default=42,
    )
    asyncio.run(main(parser.parse_args()))
