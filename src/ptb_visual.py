import logging

import plotly.graph_objects as go
from plotly.io import to_html

from src.ptb_utils import PtbScenarioRes, WordInfoRes
from src.config import AppSettings
from src.schemas import ScenarioResult, TokenEntropy

logger = logging.getLogger(__name__)


def create_highlighted_text_html(data: list[WordInfoRes], config: AppSettings) -> str:
    """
    Генерирует HTML-блок с текстом.
    Токены с высокой энтропией подсвечиваются.
    """
    entropy_threshold = config.entropy_threshold
    max_entropy_scale = config.max_entropy_scale
    html_parts = [
        '<div style="font-family: sans-serif; line-height: 1.6; border: 1px solid #ddd; padding: 15px; border-radius: 8px; background: #f9f9f9;">'
    ]
    for item in data:
        word = item["word"]
        entropy = item["entropy"]

        # Логика цвета:
        # Если энтропия < порога -> прозрачный фон.
        # Если выше -> от светло-красного до ярко-красного.
        if entropy > entropy_threshold:
            # Нормализуем альфа-канал от 0.2 до 0.8 в зависимости от силы энтропии
            alpha = min(
                0.8,
                max(
                    0.2,
                    (entropy - entropy_threshold)
                    / (max_entropy_scale - entropy_threshold),
                ),
            )
            bg_color = f"rgba(255, 0, 0, {alpha:.2f})"
            border = "1px solid rgba(255,0,0,0.3)"
        else:
            bg_color = "transparent"
            border = "none"

        # Заменяем переносы строк на <br>, чтобы HTML не ломался
        display_token = word.replace("\n", "<br>")

        # HTML span с тултипом
        span = f"""
<span class="word-span" 
      style="background-color: {bg_color}; border-bottom: {border}; cursor: help; padding: 0 2px; border-radius: 3px;"
      title="Word: '{word.strip()}'&#10;Entropy: {entropy:.3f} bits">
{display_token}
</span>
        """
        html_parts.append(span)

    html_parts.append("</div>")
    return "".join(html_parts)


def create_plotly_chart(
    data: list[WordInfoRes], title: str, config: AppSettings
) -> str:
    """Создает график Plotly для вставки в HTML."""
    words = [d["word"] for d in data]
    entropies = [d["entropy"] for d in data]

    entropy_threshold = config.entropy_threshold

    fig = go.Figure()

    # Линия энтропии
    fig = fig.add_trace(
        go.Scatter(
            x=list(range(len(words))),
            y=entropies,
            mode="lines+markers",
            name="Entropy",
            line=dict(color="#1f77b4"),
            hovertext=[f"'{t}'" for t in words],
        )
    )

    # Линия порога
    fig = fig.add_hline(
        y=entropy_threshold,
        line_dash="dash",
        line_color="red",
        annotation_text="Threshold",
    )

    fig = fig.update_layout(
        title=f"График энтропии: {title}",
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        yaxis_title="Entropy (bits)",
    )

    # Возвращаем HTML-строку графика (только div, без полного html)
    return to_html(fig, full_html=False, include_plotlyjs=True)


def generate_full_report(
    scenario_results: list[tuple[str, PtbScenarioRes]],
    filename: str,
    config: AppSettings,
):
    """Собирает всё в один красивый HTML файл."""

    # CSS стили для страницы
    full_html = """
    <html>
    <head>
        <title>LLM Entropy Analysis</title>
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; max-width: 1000px; margin: 0 auto; padding: 20px; color: #333; }
            h2 { border-bottom: 2px solid #333; padding-bottom: 10px; margin-top: 40px; }
            .section { margin-bottom: 50px; }
            .token-span:hover { outline: 2px solid #333; z-index: 10; position: relative; }
        </style>
    </head>
    <body>
    <h1>Анализ "Точек перегиба" (Inflection Points)</h1>
    <p>Красным цветом выделены слова, где модель испытывала высокую неуверенность (Internal Confusion).</p>
    """

    for idx, sc in scenario_results:
        logger.info(f"Обработка сценария: {idx}...")

        chart_html = create_plotly_chart(sc["words"], sc["name"], config)
        text_html = create_highlighted_text_html(sc["words"], config)

        full_html += f"""
        <div class="section">
            <h2>Сценарий: {sc["name"]}</h2>
            <p><strong>Prompt:</strong> <em>{sc["context"][:100]}...</em></p>
            {chart_html}
            <h3>Визуализация текста (наведите на слова):</h3>
            {text_html}
        </div>
        """

    full_html += "</body></html>"

    with open(filename, "w", encoding="utf-8") as f:
        _ = f.write(full_html)

    logger.info(f"Готово! Откройте файл {filename} в браузере.")
