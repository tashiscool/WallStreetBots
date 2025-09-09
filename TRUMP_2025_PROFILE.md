# 🇺🇸 Trump 2025 Trading Profile - Policy-Aware Strategy Configuration

> **Evidence-Based Adaptation**: Implements specific adjustments based on 2025 policy regime analysis and market dynamics.

## 📊 **Profile Overview**

The `trump_2025` profile adapts to the current political and market environment with:
- **AI infrastructure focus** (deregulation + domestic fab incentives)
- **M&A opportunity emphasis** (antitrust relaxation)
- **Policy volatility management** (tariff headline risk)
- **Israeli tech M&A scanner** (high-value acquisition targets)

### **Risk Parameters:**
- **Max Total Risk**: 55% (moderate increase from research baseline)
- **Max Position Size**: 25% (balanced for policy volatility)  
- **Data Refresh**: 20s (faster than research, slower than WSB)

---

## 🎯 **Strategy-by-Strategy Adjustments**

### **📈 EMPHASIZED STRATEGIES**

#### **1. SPX Credit Spreads (↑15% allocation)**
**Why**: Policy headlines spike IV intraday; harvest premium once path clarifies
- **Iron Condors**: 28-35 DTE, 10-15 delta shorts
- **50% take profit**, 2.2x stop loss
- **VIX filter**: Skip entries when VIX >25
- **Morning entries**: Avoid late-day policy announcements

#### **2. Momentum Weeklies (↑12% allocation)**  
**Why**: AI Action Plan + tariffs create news-driven bursts in semis/equipment
- **AI Infra Focus**: SPY/QQQ/SMH/SOXX/INTC/MU/AMAT/LRCX/NVDA/AVGO
- **After 10:00 ET entries**: Avoid pre-market policy noise
- **0-2 DTE only**: Quick in/out on clear momentum
- **50% profit target**, 2x stop

#### **3. Wheel Strategy (↑30% allocation)**
**Why**: Lighter AI regulation + domestic-fab incentives favor CSP/CC income
- **30-45 DTE**, 15-25 delta range
- **AI infra + megacap focus**: Liquid, policy-beneficiary names
- **Avoid tariff decision weeks**: Skip high-uncertainty periods

### **📊 SELECTIVE/MODIFIED STRATEGIES**

#### **4. Earnings Protection (↓8% allocation)**
**Why**: Expensive vol argues for being choosy; focus on policy exposure
- **Higher IV threshold**: 60%+ (be selective)
- **Tight RICs preferred**: When IV is very high
- **Exit pre-announce**: If IV run-up is extreme
- **Supply chain focus**: Names with tariff/export exposure

#### **5. Debit Spreads (↔15% allocation)**
**Why**: Policy tailwinds favor infra; avoid pure AI app layer
- **AI infrastructure focus**: Equipment, power, data centers
- **Avoid pure AI apps**: Most exposed to sentiment air-pockets
- **Trend confirmation required**: 40%+ trend strength
- **40% profit target**: Take partials on directional wins

#### **6. WSB Dip Bot (↓15% allocation)**
**Why**: Policy-driven dips often mean-revert; stick to liquid leaders
- **Index heavyweights only**: SPY/QQQ core holdings
- **News-resolved filter**: Wait for tariff/policy details to clarify
- **5-7 day lookback**: Slightly longer for policy clarity
- **Smaller OTM**: 2.5% vs 2.0% to reduce headline sensitivity

### **🔻 DE-EMPHASIZED STRATEGIES**

#### **7. LEAPS Tracker (↓8% allocation)**
**Why**: Tariff/export rules reduce confidence in 18-24mo outcomes
- **US-fab and infra only**: Avoid broad AI exposure
- **Entry staging required**: Stagger entries over time
- **Constrained allocation**: Max 20% vs 40% in WSB profile

#### **8. Swing Trading (↓5% allocation)**  
**Why**: Overnight headline risk elevated; news whipsaws common
- **Day-only holds**: Sub-24h maximum
- **Higher volume requirement**: 2.0x vs 1.4x spike needed
- **News catalyst required**: Volume + specific catalyst
- **Israeli tech included**: M&A speculation scanner

#### **9. Lotto Scanner (DISABLED)**
**Why**: Policy shocks create bimodal outcomes hard to handicap
- **Status**: Disabled by default
- **If enabled**: Max 1% per trade, catalyst required
- **Rationale**: Capital better allocated to structured strategies

---

## 📋 **Key Watchlists**

### **🔧 AI Infrastructure Core:**
`['SPY','QQQ','SMH','SOXX','INTC','MU','AMAT','LRCX','ACLS','NVDA','AVGO','QCOM','TXN','ADI']`

**Why**: White House AI Action Plan emphasizes deregulation, faster permitting, domestic fab support

### **🤝 M&A Beneficiaries:**
`['XLF','KRE','IBB','JETS','XLE','XLU','XLRE']`

**Why**: Antitrust agencies accepting structural remedies again; deal flow increasing

### **🇮🇱 Israeli Tech Scanner:**
`['CYBR','S','CHKP','NICE','MNDY','WIX','FROG']`

**Why**: H1'25 funding hit multi-year high; mega-deals (Wiz >$30B) show strategic value

---

## ⚡ **Policy-Specific Risk Controls**

### **📅 Event Filters:**
- **Avoid Policy Announcement Days**: Major tariff/trade announcements
- **Avoid Tariff Decision Weeks**: Known decision windows  
- **VIX Regime Filter**: Skip entries when VIX >25 (regime shifts)
- **News Resolution Required**: Wait for policy details before WSB dip entries

### **📊 Volatility Management:**
- **Morning Entry Bias**: Avoid late-day policy surprise risk
- **Tighter Profit Targets**: 40-50% vs 60%+ in other profiles
- **Policy Headline Hedging**: Built into position sizing
- **Overnight Risk Limits**: Swing trades <24h holds

---

## 💡 **Usage Example**

```python
from backend.tradingbot.production.core.production_strategy_manager import (
    ProductionStrategyManagerConfig, StrategyProfile
)

# Trump 2025 Policy-Aware Configuration
config = ProductionStrategyManagerConfig(
    alpaca_api_key=os.getenv('ALPACA_API_KEY'),
    alpaca_secret_key=os.getenv('ALPACA_SECRET_KEY'),
    paper_trading=True,
    profile=StrategyProfile.trump_2025,  # 🇺🇸 Policy-aware settings
    user_id=1,
    enable_alerts=True
)

manager = ProductionStrategyManager(config)
print(f"✅ Loaded {len(manager.strategies)}/10 strategies")
print(f"📊 Profile: {config.profile}")
print(f"🛡️ Max Risk: {config.max_total_risk:.0%}")
print(f"💰 Max Position: {config.max_position_size:.0%}")
```

---

## 🔄 **Profile Comparison**

| Setting | Research 2024 | WSB 2025 | **Trump 2025** | Impact |
|---------|---------------|----------|----------------|---------|
| **Max Risk** | 50% | 65% | **55%** | Moderate for policy volatility |
| **Max Position** | 20% | 30% | **25%** | Balanced approach |
| **Data Refresh** | 30s | 10s | **20s** | Policy-aware timing |
| **SPX Spreads** | 4% | 12% | **15%** | Harvest policy vol premium |
| **Momentum** | 5% | 10% | **12%** | AI infra focus |
| **Wheel** | 20% | 42% | **30%** | Policy beneficiaries |
| **LEAPS** | 3% | 16% | **8%** | Reduced long-term uncertainty |
| **Lotto** | 1% | 4% | **0%** | Hard to handicap policy shocks |

---

## 📚 **Source Attribution**

This profile implements evidence-based adjustments from:
- **White House AI Action Plan** (deregulation, domestic fab incentives)
- **Reuters** (100%+ semiconductor tariffs, China export arrangements)  
- **Financial Times** (AI bubble concerns, stretched multiples)
- **McDermott/ABA** (antitrust relaxation, structural remedies)
- **Bloomberg/Times of Israel** (Israeli tech M&A surge, $13.4B exits)

**Key Insight**: Policy creates both opportunities (M&A, AI infra) and risks (volatility, headline sensitivity). This profile balances both while maintaining disciplined risk management.

---

<div align="center">

## 🎯 **Ready for 2025 Policy Environment**

**Emphasizes evidence-based opportunities while managing headline risk**

[Back to Main README](README.md) | [Strategy Configurations](backend/tradingbot/production/core/production_strategy_manager.py)

</div>