import argparse


parser = argparse.ArgumentParser()
_ = parser.add_argument(
    "-m",
    "--model",
    type=str,
    required=True,
    help="id модели",
)
_ = parser.add_argument(
    "-r",
    "--report",
    type=str,
    required=True,
    help="Путь до файла с отчётом",
)
_ = parser.add_argument(
    "-c",
    "--config",
    type=str,
    required=False,
    default="./congig.json",
    help="Путь до файла с конфигурацией",
)
