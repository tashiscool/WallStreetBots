import json

from django.contrib.auth.models import User
from django.test import Client, TestCase
from django.urls import reverse

from .models import Company


class CompanyApiTests(TestCase):
    def setUp(self):
        self.client = Client()
        test_password = "testpass123"  # noqa: S105
        self.user = User.objects.create_user(
            username="tradingbot_user",
            password=test_password,
        )
        self.client.login(username="tradingbot_user", password=test_password)

    def test_create_company(self):
        response = self.client.post(
            reverse("company_create"),
            data=json.dumps({"ticker": "AAPL", "name": "Apple Inc."}),
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 201)
        self.assertTrue(Company.objects.filter(ticker="AAPL").exists())

    def test_patch_company(self):
        Company.objects.create(ticker="AAPL", name="Apple")

        response = self.client.patch(
            reverse("company_patch", kwargs={"ticker": "AAPL"}),
            data=json.dumps({"name": "Apple Inc."}),
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            Company.objects.get(ticker="AAPL").name,
            "Apple Inc.",
        )
