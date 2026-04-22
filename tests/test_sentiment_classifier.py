import os
import tempfile
import unittest

from sentiment_classifier import SentimentClassifier


class DummyVectorizer:
    def transform(self, texts):
        return texts


class DummyModel:
    def predict(self, text_vec):
        return ["positive"]

    def predict_proba(self, text_vec):
        return [[0.85, 0.10, 0.05]]


class SentimentClassifierTests(unittest.TestCase):
    def setUp(self):
        self.classifier = SentimentClassifier()
        self.classifier.model = DummyModel()
        self.classifier.vectorizer = DummyVectorizer()

    def test_classify_text_requires_loaded_model(self):
        c = SentimentClassifier()
        with self.assertRaises(ValueError):
            c.classify_text("test")

    def test_classify_text_returns_prediction_and_confidence(self):
        sentiment, confidence = self.classifier.classify_text("great result")
        self.assertEqual(sentiment, "positive")
        self.assertAlmostEqual(confidence, 0.85, places=3)

    def test_classify_dataset_adds_sentiment_fields(self):
        data = [
            {"title": "Post 1", "selftext": "Body 1"},
            {"cleaned_text": "already cleaned"},
        ]
        output = self.classifier.classify_dataset(data)
        self.assertEqual(len(output), 2)
        self.assertIn("sentiment", output[0])
        self.assertIn("sentiment_confidence", output[0])
        self.assertEqual(output[1]["sentiment"], "positive")

    def test_generate_sentiment_report_returns_counts(self):
        classified_data = [
            {
                "sentiment": "positive",
                "sentiment_confidence": 0.91,
                "title": "A",
                "cleaned_text": "good news",
                "subreddit": "farming",
            },
            {
                "sentiment": "negative",
                "sentiment_confidence": 0.88,
                "title": "B",
                "cleaned_text": "bad news",
                "subreddit": "farming",
            },
            {
                "sentiment": "neutral",
                "sentiment_confidence": 0.75,
                "title": "C",
                "cleaned_text": "mixed news",
                "subreddit": "technology",
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = os.path.join(tmpdir, "report.txt")
            counts, percentages = self.classifier.generate_sentiment_report(
                classified_data,
                output_file=out_file,
            )
            self.assertTrue(os.path.exists(out_file))
            self.assertEqual(counts["positive"], 1)
            self.assertAlmostEqual(percentages["positive"], 100 / 3, places=2)


if __name__ == "__main__":
    unittest.main()
