import json

from django.contrib.auth.models import User
from django.test import Client, TestCase
from django.urls import reverse

from backend.tradingbot.models import Company


class TestCompanyApi(TestCase):
    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user(
            username="company_api_user",
            password="testpass123",
        )
        self.client.login(username="company_api_user", password="testpass123")

    def test_create_company(self):
        response = self.client.post(
            reverse("company_create"),
            data=json.dumps({"ticker": "AAPL", "name": "Apple Inc."}),
            content_type="application/json",
        )

        assert response.status_code == 201
        assert Company.objects.filter(ticker="AAPL", name="Apple Inc.").exists()

    def test_create_company_rejects_invalid_ticker(self):
        response = self.client.post(
            reverse("company_create"),
            data=json.dumps({"ticker": "AAPL!", "name": "Apple Inc."}),
            content_type="application/json",
        )

        assert response.status_code == 400

    def test_patch_company_name(self):
        Company.objects.create(ticker="MSFT", name="Microsoft")

        response = self.client.patch(
            reverse("company_patch", kwargs={"ticker": "MSFT"}),
            data=json.dumps({"name": "Microsoft Corp."}),
            content_type="application/json",
        )

        assert response.status_code == 200
        assert Company.objects.get(ticker="MSFT").name == "Microsoft Corp."

    def test_patch_company_requires_name(self):
        Company.objects.create(ticker="MSFT", name="Microsoft")

        response = self.client.patch(
            reverse("company_patch", kwargs={"ticker": "MSFT"}),
            data=json.dumps({}),
            content_type="application/json",
        )

        assert response.status_code == 400
