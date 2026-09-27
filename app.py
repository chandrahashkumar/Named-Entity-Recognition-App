from __future__ import annotations

from functools import lru_cache

import gradio as gr
from transformers import pipeline


@lru_cache(maxsize=1)
def get_ner_pipeline():
    return pipeline(
        task="token-classification",
        model="dslim/bert-base-NER",
        aggregation_strategy="simple",
    )


def recognize_entities(text: str):
    if not text or not text.strip():
        return []

    try:
        results = get_ner_pipeline()(text)
    except OSError as exc:
        return {"error": f"Model load failed: {exc}"}

    return [
        {
            "entity": item["entity_group"],
            "word": item["word"],
            "start": item["start"],
            "end": item["end"],
            "score": round(float(item["score"]), 4),
        }
        for item in results
    ]


app = gr.Interface(
    fn=recognize_entities,
    inputs=gr.Textbox(
        lines=4,
        label="Input text",
        placeholder="Enter text to extract named entities",
    ),
    outputs=gr.JSON(label="Recognized entities"),
    title="Named Entity Recognition App",
    description="NER app powered by Hugging Face Transformers and dslim/bert-base-NER.",
)


if __name__ == "__main__":
    app.launch()
