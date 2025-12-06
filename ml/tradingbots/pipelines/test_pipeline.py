import unittest

from .pipeline import Pipeline


# Rename classes to avoid pytest collection (they're helper classes, not test classes)
class PipelineSubClass1(Pipeline):
    def pipeline(self):
        return {"AAPL": 5, "MSFT": 2, "TSLA": 2}


class PipelineSubClass2(Pipeline):
    def pipeline(self):
        return {"AAPL": 5, "MSFT": 4}


class PipelineSubClass3(Pipeline):
    def pipeline(self):
        return {"MSFT": 9}


class PipelineTestCase(unittest.TestCase):
    portfolio = {"cash": 1000, "stocks": {"AAPL": 3, "MSFT": 6}}

    def test_case_1(self):
        pipeline = PipelineSubClass1("p1", self.portfolio)
        actions = pipeline.rebalance()
        actions = [a.__dict__() for a in actions]
        self.assertIn(
            {
                "order_type": "M",
                "transaction_type": "B",
                "ticker": "TSLA",
                "quantity": 2,
            },
            actions,
        )
        self.assertIn(
            {
                "order_type": "M",
                "transaction_type": "B",
                "ticker": "AAPL",
                "quantity": 2,
            },
            actions,
        )
        self.assertIn(
            {
                "order_type": "M",
                "transaction_type": "S",
                "ticker": "MSFT",
                "quantity": 4,
            },
            actions,
        )

    def test_case_2(self):
        pipeline = PipelineSubClass2("p2", self.portfolio)
        actions = pipeline.rebalance()
        actions = [a.__dict__() for a in actions]
        self.assertIn(
            {
                "order_type": "M",
                "transaction_type": "B",
                "ticker": "AAPL",
                "quantity": 2,
            },
            actions,
        )
        self.assertIn(
            {
                "order_type": "M",
                "transaction_type": "S",
                "ticker": "MSFT",
                "quantity": 2,
            },
            actions,
        )

    def test_case_3(self):
        pipeline = PipelineSubClass3("p3", self.portfolio)
        actions = pipeline.rebalance()
        actions = [a.__dict__() for a in actions]
        self.assertIn(
            {
                "order_type": "M",
                "transaction_type": "S",
                "ticker": "AAPL",
                "quantity": 3,
            },
            actions,
        )
        self.assertIn(
            {
                "order_type": "M",
                "transaction_type": "B",
                "ticker": "MSFT",
                "quantity": 3,
            },
            actions,
        )


if __name__ == "__main__":
    unittest.main()
