from django.apps import AppConfig


def create_portfolio_dictionary(user):
    from .models import StockInstance
    from backend.tradingbot.synchronization import sync_alpaca
    sync_alpaca(user)
    portfolio = user.portfolio
    result = {'cash': float(portfolio.cash)}
    stock_instances = StockInstance.objects.filter(portfolio=portfolio)
    # structure to return
    # {
    #   cash: float,
    #   stocks: {
    #       ticker: qty
    #   }
    # }
    result['stocks'] = {}
    for stock_instances in stock_instances:
        result['stocks'][stock_instances.stock.company.ticker] = float(stock_instances.quantity)
    return result


def start_pipelines():
    try:
        from django.contrib.auth.models import User
        from backend.tradingbot.apiutility import place_general_order
        from backend.tradingbot.synchronization import sync_alpaca
        # from ml.tradingbots.pipelines.monte_carlo_w_ma import MonteCarloMovingAveragePipline
        from ml.tradingbots.trader import MonteCarloMASharpeRatioStrategy
        # TODO: dynamically get the right strategy
        # rebalancing_strategies = {
        #     "monte_carlo": MonteCarloMASharpeRatioStrategy,
        #     "hmm": None
        # }
        users_to_actions = {}
        for user in User.objects.all():
            if user.portfolio:
                strat = MonteCarloMASharpeRatioStrategy("Name")
                actions = strat.get_actions(create_portfolio_dictionary(user))
                users_to_actions[user] = actions
        for user, actions in users_to_actions.items():
            for action in actions:
                place_general_order(
                    user=user,
                    user_details=sync_alpaca(user),
                    ticker=action.ticker,
                    quantity=action.quantity,
                    transaction_type=action.transaction_type,
                    order_type=action.order_type,
                    time_in_force="day"
                )
    except ImportError as e:
        # Silently fail if dependencies are not available
        pass


class TradingbotConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'backend.tradingbot'

    def ready(self):
        try:
            from apscheduler.schedulers.background import BackgroundScheduler
            scheduler = BackgroundScheduler()
            scheduler.add_job(start_pipelines, 'interval', seconds=60*60*24)
            scheduler.start()
        except ImportError:
            # Silently fail if APScheduler is not available
            pass
