# Wallstreetbots Architecture Overview
## Project: wallstreetbots
## Date: 2026-02-20

## System Context
# 🚀 WallStreetBots - Institutional-Grade Algorithmic Trading System

<div align="center">

## **Production-Ready Trading Platform with Advanced Risk Management**
### *Sophisticated WSB-style strategies • Institutional risk controls • Real broker integration*

**✅ 10+ Complete Trading Strategies** • **✅ Advanced VaR/CVaR Risk Models** • **✅ ML Risk Agents** • **✅ Multi-Asset Support**

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://python.org)
[![Django](https://img.shields.io/badge/Django-4.2+-green.svg)](https://djangoproject.com)
[![Tests](https://img.shields.io/badge/Tests-5500+-brightgreen.svg)](#testing)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

A **comprehensive, institutional-grade trading system** implementing WSB-style strategies with **sophisticated risk management**, **real-time monitoring**, and **production-ready architecture**. 

**🎯 What It Does:** Automatically finds trading opportunities, places trades, manages risk, and tracks performance - like having a professional trader working for you 24/7.

**🛡️ Safety First:** Built-in risk management protects your capital with multiple safety layers including position limits, stop losses, and circuit breakers.

**📚 New to Trading?** Start with our [5-Minute Quick Start](docs/QUICK_START.md) or read [How It Works](docs/HOW_IT_WORKS.md) for a simple explanation!

## 🏆 **Key Capabilities**

### **📈 Trading Strategies (10+ Complete)**
- **WSB Dip Bot** - Momentum-based dip buying with volume confirmation
- **Earnings Protection** - Options-based ea

## Layer Map
- presentation: tbd
- application: backend
- domain: test_models
- infrastructure: config; docker; ops; scripts

## Entry Points
- run.py
- manage.py
- run_wallstreetbots.py

## External Dependencies
- Django
- djangorestframework
- dj-database-url
- whitenoise
- social-auth-app-django
- python-jose
- authlib
- psycopg2-binary
- asgiref
- requests
- httpx
- aiohttp
- numpy
- pandas
- scipy

## External Services
- postgres
- github

## Notes
- Source repo: /Users/tkhan/projects/WallStreetBots
