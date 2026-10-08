import unittest
from modele import entrainer


class TestClassification(unittest.TestCase):
    def test_message_spam(self):
        modele = entrainer()
        self.assertEqual(modele.predict(["cadeau gratuit"])[0], "spam")

    def test_message_normal(self):
        modele = entrainer()
        self.assertEqual(modele.predict(["reunion de projet"])[0], "normal")


if __name__ == "__main__":
    unittest.main()
