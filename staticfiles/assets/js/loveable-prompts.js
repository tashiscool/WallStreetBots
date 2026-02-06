/**
 * WallStreetBots Loveable Prompts Library
 *
 * Universal tooltip definitions and contextual implication boxes
 * that help users understand trading concepts and system behavior.
 */

// =============================================================================
// HELP LIBRARY - Universal Tooltip Definitions
// =============================================================================

const HELP_LIBRARY = {
  // Trading Basics
  paper_trading: {
    title: "Paper Trading",
    definition: "Simulated trading with fake money ($100,000) to practice without risk.",
    example: "Like a flight simulator - learn the controls before flying the real plane."
  },
  live_trading: {
    title: "Live Trading",
    definition: "Real trades with real money through your connected brokerage account.",
    example: "Your actual money is at stake. Start small and scale up as you gain confidence."
  },
  position: {
    title: "Position",
    definition: "An open trade - stock or options you currently own or have sold short.",
    example: "If you bought 100 shares of AAPL, that's one position. Multiple contracts of the same option also count as one position."
  },
  stop_loss: {
    title: "Stop Loss",
    definition: "An automatic sell order that limits how much you can lose on a trade.",
    example: "Buy at $100, stop loss at $95 = max loss of 5%. The system sells automatically if price hits $95."
  },
  take_profit: {
    title: "Take Profit",
    definition: "An automatic sell order that locks in gains at a target price.",
    example: "Buy at $100, take profit at $110 = you lock in 10% gain automatically when price hits $110."
  },

  // Risk Metrics
  var: {
    title: "Value at Risk (VaR)",
    definition: "Maximum expected loss over a time period at a given confidence level.",
    example: "95% VaR of $2,000 means: on 95 out of 100 days, you won't lose more than $2,000."
  },
  drawdown: {
    title: "Drawdown",
    definition: "The decline from a peak to a trough. How far you've fallen from your best point.",
    example: "If portfolio hit $110k then dropped to $95k, that's a 13.6% drawdown."
  },
  max_drawdown: {
    title: "Maximum Drawdown",
    definition: "The largest peak-to-trough decline in your portfolio's history.",
    example: "Used to measure worst-case scenarios. Under -20% is generally acceptable for aggressive strategies."
  },
  circuit_breaker: {
    title: "Circuit Breaker",
    definition: "Automatic trading halt when risk limits are hit - like electrical fuses.",
    example: "If daily loss hits 6%, trading pauses automatically. Prevents compounding losses."
  },

  // Performance Metrics
  sharpe_ratio: {
    title: "Sharpe Ratio",
    definition: "Return per unit of risk taken. Higher = better risk-adjusted returns.",
    example: ">1.0 is good, >2.0 is excellent. Hedge funds typically target 1.5+."
  },
  sortino_ratio: {
    title: "Sortino Ratio",
    definition: "Like Sharpe, but only penalizes downside volatility. Ignores upside swings.",
    example: "Higher than Sharpe means your gains are bigger than your drops. Good sign!"
  },
  alpha: {
    title: "Alpha",
    definition: "Excess return beyond what market exposure (beta) explains. Your 'edge'.",
    example: "+4% alpha means you're beating the market by 4% through skill, not just riding the market up."
  },
  beta: {
    title: "Beta",
    definition: "How much you move relative to the market (S&P 500).",
    example: "Beta 1.2 = when market moves 1%, you move 1.2%. Higher = more volatile."
  },
  win_rate: {
    title: "Win Rate",
    definition: "Percentage of trades that make money.",
    example: "50% = break-even threshold, 60% = good, 70%+ = excellent. But size matters too!"
  },
  profit_factor: {
    title: "Profit Factor",
    definition: "Gross profits divided by gross losses.",
    example: "1.8 = you make $1.80 for every $1 you lose. >1.5 is good, >2.0 is excellent."
  },

  // Options Greeks
  delta: {
    title: "Delta",
    definition: "How much option price changes per $1 stock move. Also approximates probability of profit.",
    example: "Delta 0.50 = option gains $0.50 per $1 stock rise. Also ~50% chance of expiring ITM."
  },
  gamma: {
    title: "Gamma",
    definition: "How fast delta changes. Acceleration of your position.",
    example: "High gamma = explosive gains/losses as stock moves. Like driving faster - more responsive but riskier."
  },
  theta: {
    title: "Theta",
    definition: "Time decay - how much value options lose each day.",
    example: "Theta -$5 = option loses $5 per day just from time passing. Sellers collect it, buyers pay it."
  },
  vega: {
    title: "Vega",
    definition: "Sensitivity to volatility changes.",
    example: "Vega $10 = option gains $10 for every 1% increase in implied volatility."
  },

  // Strategy Types
  wheel_strategy: {
    title: "Wheel Strategy",
    definition: "Sell puts until assigned, then sell calls until called away. Income-focused.",
    example: "Like being a landlord for stocks - collect 'rent' (premium) while willing to own or sell shares."
  },
  momentum: {
    title: "Momentum Trading",
    definition: "Buy things going up, sell things going down. Trend following.",
    example: "If NVDA is up 5% this week, momentum says it's likely to keep going. Ride the wave."
  },
  mean_reversion: {
    title: "Mean Reversion",
    definition: "Bet that extreme moves will reverse back to average.",
    example: "Stock down 15% in a day with no news? Mean reversion bets it bounces back."
  }
};

// =============================================================================
// IMPLICATION BOXES - Contextual Guidance System
// =============================================================================

const IMPLICATION_BOXES = {
  // Success/Achievement Messages
  first_trade: {
    title: "Your First Trade is Live!",
    body: "The bot just placed your first real trade. Don't panic - this is what it's supposed to do. Check your Alpaca dashboard to see the position.",
    type: "success",
    icon: "fa-rocket"
  },
  profitable_week: {
    title: "Profitable Week!",
    body: "You ended the week in the green. The system is working. Remember: consistency beats occasional home runs.",
    type: "success",
    icon: "fa-trophy"
  },
  strategy_milestone: {
    title: "Strategy Milestone Reached",
    body: "This strategy has now completed 30+ trades - enough for statistically meaningful analysis. Check the Analytics page for insights.",
    type: "info",
    icon: "fa-chart-line"
  },

  // Warning/Caution Messages
  losing_streak: {
    title: "Losing Streak - Stay Calm",
    body: "3+ losses in a row is normal - even the best strategies have drawdowns. Review positions but don't panic-sell. If system hasn't triggered circuit breakers, it's within normal parameters.",
    type: "warning",
    icon: "fa-hand-paper"
  },
  high_volatility: {
    title: "High Volatility Detected",
    body: "VIX is elevated (>25). Position sizes have been automatically reduced. This is protective - the system is being more cautious.",
    type: "warning",
    icon: "fa-exclamation-triangle"
  },
  approaching_limit: {
    title: "Approaching Risk Limit",
    body: "You're at 80% of a risk limit. The system will automatically pause that activity if the limit is hit. This is working as designed.",
    type: "warning",
    icon: "fa-tachometer-alt"
  },

  // Circuit Breaker Messages
  circuit_breaker_triggered: {
    title: "Circuit Breaker Activated",
    body: "Trading has been automatically paused because a risk limit was hit. This is protection working as intended. Review the Risk page, take a break, and restart when ready.",
    type: "danger",
    icon: "fa-bolt"
  },
  daily_loss_limit: {
    title: "Daily Loss Limit Hit",
    body: "You've reached your maximum daily loss threshold. All trading is paused until tomorrow. Use this time to review what happened - was it market conditions or strategy issues?",
    type: "danger",
    icon: "fa-stop-circle"
  },

  // Transition Messages
  paper_to_live_transition: {
    title: "Ready for Live Trading?",
    body: "Before switching: 1) Review your paper trading results, 2) Start with 25% of your intended capital, 3) Keep the same settings that worked in paper, 4) Have realistic expectations - real slippage exists.",
    type: "info",
    icon: "fa-exchange-alt",
    checklist: [
      "Paper trading results reviewed and satisfactory",
      "Risk settings configured appropriately",
      "Starting with reduced capital for first week",
      "Brokerage account verified and funded"
    ]
  },
  new_strategy_activated: {
    title: "New Strategy Activated",
    body: "Give it at least 2-3 weeks and 20+ trades before judging performance. Early volatility is normal. Check back in a week for meaningful data.",
    type: "info",
    icon: "fa-play-circle"
  },

  // Helpful Context
  market_closed: {
    title: "Markets Are Closed",
    body: "US stock markets are closed (weekends, holidays, or after hours). The bot monitors but can't trade. Perfect time to review your strategy and settings.",
    type: "info",
    icon: "fa-moon"
  },
  first_week_check: {
    title: "First Week Complete!",
    body: "You've been running for a week. This is a good time to: 1) Review the Analytics page, 2) Check if actual behavior matches expectations, 3) Fine-tune settings if needed. Don't make major changes based on just one week though.",
    type: "info",
    icon: "fa-calendar-check"
  }
};

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================

/**
 * Show an implication box at a target element
 * @param {string} boxId - Key from IMPLICATION_BOXES
 * @param {string} targetSelector - CSS selector for where to insert the box
 * @param {string} position - 'before', 'after', or 'replace'
 */
function showImplicationBox(boxId, targetSelector, position = 'after') {
  const box = IMPLICATION_BOXES[boxId];
  if (!box) {
    console.warn(`Implication box '${boxId}' not found`);
    return;
  }

  const typeColors = {
    success: 'rgba(45,206,137,0.1)',
    warning: 'rgba(251,99,64,0.1)',
    danger: 'rgba(245,54,92,0.1)',
    info: 'rgba(17,205,239,0.1)'
  };

  const iconColors = {
    success: 'text-success',
    warning: 'text-warning',
    danger: 'text-danger',
    info: 'text-info'
  };

  let checklistHtml = '';
  if (box.checklist) {
    checklistHtml = '<ul class="mt-2 mb-0 ps-3">' +
      box.checklist.map(item => `<li class="text-sm">${item}</li>`).join('') +
      '</ul>';
  }

  const html = `
    <div class="implication-box-dynamic mb-3" style="background: linear-gradient(to right, ${typeColors[box.type]}, rgba(255,255,255,0.02)); border-radius: 8px; padding: 16px; border-left: 4px solid;">
      <h6 class="mb-2"><i class="fas ${box.icon} ${iconColors[box.type]} me-2"></i>${box.title}</h6>
      <p class="text-sm text-secondary mb-0">${box.body}</p>
      ${checklistHtml}
      <button class="btn btn-link btn-sm p-0 mt-2 dismiss-implication" onclick="this.parentElement.remove()">
        <i class="fas fa-times me-1"></i>Dismiss
      </button>
    </div>
  `;

  const target = document.querySelector(targetSelector);
  if (target) {
    if (position === 'before') {
      target.insertAdjacentHTML('beforebegin', html);
    } else if (position === 'after') {
      target.insertAdjacentHTML('afterend', html);
    } else if (position === 'replace') {
      target.innerHTML = html;
    }
  }
}

/**
 * Get help text for a term
 * @param {string} term - Key from HELP_LIBRARY
 * @returns {object} The help definition object
 */
function getHelpText(term) {
  return HELP_LIBRARY[term] || null;
}

/**
 * Initialize dynamic help tooltips
 * Looks for elements with data-help="term" and adds tooltips
 */
function initHelpTooltips() {
  const helpElements = document.querySelectorAll('[data-help]');
  helpElements.forEach(el => {
    const term = el.getAttribute('data-help');
    const help = HELP_LIBRARY[term];
    if (help) {
      el.setAttribute('data-bs-toggle', 'tooltip');
      el.setAttribute('data-bs-html', 'true');
      el.setAttribute('title', `<strong>${help.title}</strong><br>${help.definition}<br><em>${help.example}</em>`);
      el.style.cursor = 'help';
      el.classList.add('help-term');

      // Add a subtle indicator
      if (!el.querySelector('.help-indicator')) {
        const indicator = document.createElement('i');
        indicator.className = 'fas fa-question-circle text-secondary opacity-6 ms-1 help-indicator';
        indicator.style.fontSize = '0.8em';
        el.appendChild(indicator);
      }
    }
  });

  // Re-initialize Bootstrap tooltips
  var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
  tooltipTriggerList.forEach(function(tooltipTriggerEl) {
    new bootstrap.Tooltip(tooltipTriggerEl, {
      html: true
    });
  });
}

// =============================================================================
// AUTO-INITIALIZATION
// =============================================================================

document.addEventListener('DOMContentLoaded', function() {
  // Initialize help tooltips on page load
  initHelpTooltips();

  // Check for contextual triggers (can be expanded based on app state)
  // These would typically be triggered by the backend passing state

  // Example: Show first trade box if flagged
  if (window.wsb_show_first_trade) {
    showImplicationBox('first_trade', '.main-content', 'before');
  }

  // Example: Show losing streak warning if flagged
  if (window.wsb_losing_streak) {
    showImplicationBox('losing_streak', '.page-header, .card:first-child', 'after');
  }

  // Example: Show circuit breaker message if flagged
  if (window.wsb_circuit_breaker_triggered) {
    showImplicationBox('circuit_breaker_triggered', '.main-content', 'before');
  }
});

// Export for use in other scripts
window.WSB_HELP = HELP_LIBRARY;
window.WSB_IMPLICATIONS = IMPLICATION_BOXES;
window.showImplicationBox = showImplicationBox;
window.getHelpText = getHelpText;
window.initHelpTooltips = initHelpTooltips;
