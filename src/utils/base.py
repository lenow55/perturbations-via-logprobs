import json
import logging
from logging import config as log_config_m

from httpx import AsyncClient, Timeout
from openai import AsyncOpenAI

from src.config import AppSettings, LLMConfig

logger = logging.getLogger(__name__)


def configure_logging(config: AppSettings):
    with open(config.logging_conf_file) as l_f:
        logging_config_dict = json.loads(l_f.read())
        log_config_m.dictConfig(logging_config_dict)


def create_openai_client(config: LLMConfig) -> AsyncOpenAI:
    if config.proxy_url:
        http_client = AsyncClient(proxy="socks5h://localhost:10808")
    else:
        http_client = AsyncClient()

    client = AsyncOpenAI(
        api_key=config.api_key.get_secret_value(),
        base_url=config.base_url,
        timeout=Timeout(config.timeout),
        http_client=http_client,
    )
    return client
