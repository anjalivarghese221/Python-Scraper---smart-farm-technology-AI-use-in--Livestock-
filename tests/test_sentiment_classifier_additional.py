import json
import os
import pickle
import tempfile
import unittest
from unittest.mock import patch

import sentiment_classifier as sc_module
from sentiment_classifier import SentimentClassifier


class DummyVectorizer:
    def transform(self, texts):
        return texts


class DummyModel:
    def predict(self, text_vec):
        return ["positive"]

    def predict_proba(self, text_vec):
        return [[0.85, 0.10, 0.05]]


class SentimentClassifierAdditionalTests(unittest.TestCase):
    def test_load_model_raises_when_files_missing(self):
        c = SentimentClassifier("missing_model.pkl", "missing_vec.pkl")
        with self.assertRaises(FileNotFoundError):
            c.load_model()

    def test_load_model_success(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.pkl")
            vec_path = os.path.join(tmpdir, "vec.pkl")

            with open(model_path, "wb") as f:
                pickle.dump({"kind": "model"}, f)
            with open(vec_path, "wb") as f:
                pickle.dump({"kind": "vectorizer"}, f)

            c = SentimentClassifier(model_path, vec_path)
            c.load_model()

            self.assertEqual(c.model["kind"], "model")
            self.assertEqual(c.vectorizer["kind"], "vectorizer")

    def test_classify_dataset_progress_line_hits_at_50(self):
        c = SentimentClassifier()
        c.model = DummyModel()
        c.vectorizer = DummyVectorizer()

        data = [{"title": f"Post {i}", "selftext": "Body"} for i in range(50)]
        output = c.classify_dataset(data)
        self.assertEqual(len(output), 50)
        self.assertTrue(all("sentiment" in x for x in output))

    def test_generate_sentiment_report_handles_unknown_subreddit(self):
        c = SentimentClassifier()
        classified_data = [
            {
                "sentiment": "positive",
                "sentiment_confidence": 0.95,
                "title": "Known",
                "cleaned_text": "known text",
                "subreddit": "farming",
            },
            {
                "sentiment": "neutral",
                "sentiment_confidence": 0.60,
                "title": "Unknown",
                "cleaned_text": "unknown text",
                "subreddit": "unknown",
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = os.path.join(tmpdir, "report.txt")
            counts, _ = c.generate_sentiment_report(classified_data, output_file=out_file)
            self.assertEqual(counts["positive"], 1)
            self.assertEqual(counts["neutral"], 1)
            self.assertTrue(os.path.exists(out_file))

    def test_main_preprocessed_wrapped_data_branch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                with open("preprocessed_data.json", "w", encoding="utf-8") as f:
                    json.dump({"data": [{"title": "A", "selftext": "B"}]}, f)

                fake_classified = [
                    {
                        "title": "A",
                        "selftext": "B",
                        "sentiment": "positive",
                        "sentiment_confidence": 0.9,
                    }
                ]

                with patch.object(sc_module.SentimentClassifier, "load_model", return_value=None), \
                     patch.object(sc_module.SentimentClassifier, "classify_dataset", return_value=fake_classified), \
                     patch.object(sc_module.SentimentClassifier, "generate_sentiment_report", return_value=({}, {})):
                    sc_module.main()

                self.assertTrue(os.path.exists("classified_sentiment_data.json"))
            finally:
                os.chdir(old_cwd)

    def test_main_cleaned_missing_branch_returns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                with patch.object(sc_module.SentimentClassifier, "load_model", return_value=None):
                    sc_module.main()
                self.assertFalse(os.path.exists("classified_sentiment_data.json"))
            finally:
                os.chdir(old_cwd)

    def test_main_exception_branch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                with patch.object(sc_module.SentimentClassifier, "load_model", side_effect=RuntimeError("boom")):
                    sc_module.main()
            finally:
                os.chdir(old_cwd)


if __name__ == "__main__":
    unittest.main()
