from statistics import mean

from transformers import pipeline


class SentimentModel:
    _pipeline = None

    def __init__(self):
        if SentimentModel._pipeline is None:
            SentimentModel._pipeline = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
            )
        self.model = SentimentModel._pipeline

    def analyze_text(self, text: str):
        if not text or not text.strip():
            return {"label": "NEUTRAL", "score": 0.0}

        result = self.model(text[:512])[0]
        return {
            "label": result["label"],
            "score": float(result["score"]),
        }

    def analyze_batch(self, texts: list[str]):
        if not texts:
            return {
                "score": 0.0,
                "count": 0,
                "article_scores": [],
            }

        signed_scores = []
        for text in texts:
            result = self.analyze_text(text)
            if result["label"] == "POSITIVE":
                signed_scores.append(result["score"])
            elif result["label"] == "NEGATIVE":
                signed_scores.append(-result["score"])
            else:
                signed_scores.append(0.0)

        final_score = mean(signed_scores)
        return {
            "score": round(final_score, 4),
            "count": len(texts),
            "article_scores": [round(score, 4) for score in signed_scores],
        }
