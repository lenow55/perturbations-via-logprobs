import argparse


parser = argparse.ArgumentParser()
_ = parser.add_argument(
    "-c",
    "--config",
    type=str,
    required=False,
    default="./congig.json",
    help="Путь до файла с конфигурацией",
)
