from __future__ import annotations
from ...core.interface import BrokerDriver
from ...core.schemas import OrderRequest, OrderResponse, BrokerCapabilities, Funds, Position, Quote
from ...core.enums import Exchange
from typing import List, Any, Dict
import csv
from datetime import datetime

import os
from ..zerodha.driver import ZerodhaDriver
from .portfolio import Portfolio
from notifiers.telegram import send_telegram_message
import uuid
import logging

class PaperBroker(BrokerDriver):
    """
    A simulated broker for paper trading.
    """

    def __init__(self, live_broker: BrokerDriver):
        super().__init__()
        self.live_broker = live_broker

        commission_rate = float(os.getenv("PAPER_COMMISSION_RATE", "0.001"))
        slippage_rate = float(os.getenv("PAPER_SLIPPAGE_RATE", "0.0005"))
        initial_cash = float(os.getenv("PAPER_INITIAL_CASH", "100000.0"))

        self.portfolio = Portfolio(initial_cash)
        self.trade_log_file = "paper_trades.csv"
        self.pnl_log_file = "pnl_log.csv"

        if not os.path.exists(self.trade_log_file):
            with open(self.trade_log_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "symbol", "transaction_type", "quantity", "price", "commission"])

        if not os.path.exists(self.pnl_log_file):
            with open(self.pnl_log_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "realized_pnl", "unrealized_pnl", "total_pnl"])

        self.capabilities = BrokerCapabilities(
            supports_historical=False,
            supports_quotes=True,
            supports_funds=True,
            supports_positions=True,
            supports_place_order=True,
            supports_modify_order=False,
            supports_cancel_order=False,
            supports_tradebook=False,
            supports_orderbook=False,
            supports_websocket=False,
            supports_order_websocket=False,
            supports_master_contract=False,
            supports_option_chain=False,
            supports_gtt=False,
            supports_bracket_order=False,
            supports_cover_order=False,
            supports_multileg_order=False,
            supports_basket_orders=False,
        )
        self._commission_rate = commission_rate
        self._slippage_rate = slippage_rate

    def get_funds(self) -> Funds:
        return Funds(equity=self.portfolio.cash, available_cash=self.portfolio.cash, used_margin=0.0, net=self.portfolio.cash, raw={})

    def get_positions(self) -> List[Position]:
        return list(self.portfolio.positions.values())

    def place_order(self, request: OrderRequest) -> OrderResponse:
        # Simulate order execution
        quote = self.get_quote(f"{request.exchange.value}:{request.symbol}")
        if quote.last_price == 0:
            return OrderResponse(status="error", order_id=None, message="No market price available")

        # Simulate slippage
        slippage = quote.last_price * self._slippage_rate
        fill_price = quote.last_price + slippage if request.transaction_type.value == "BUY" else quote.last_price - slippage

        # Simulate commission
        commission = fill_price * request.quantity * self._commission_rate

        # Update portfolio
        self.portfolio.update_position(request, fill_price, commission)

        timestamp = datetime.now()
        # Log the trade
        with open(self.trade_log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp.isoformat(),
                request.symbol,
                request.transaction_type.value,
                request.quantity,
                fill_price,
                commission
            ])

        # Log P&L
        # TODO: This re-fetches quotes for all positions after every trade, which could lead to
        # a high number of API calls for a portfolio with many open positions.
        # For a more optimized approach, consider a separate process that periodically updates all quotes.
        quotes = {pos.symbol: self.get_quote(f"{pos.exchange.value}:{pos.symbol}") for pos in self.portfolio.positions.values()}
        unrealized_pnl = self.portfolio.get_unrealized_pnl(quotes)
        total_pnl = self.portfolio.get_total_pnl(quotes)
        with open(self.pnl_log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp.isoformat(),
                self.portfolio.realized_pnl,
                unrealized_pnl,
                total_pnl
            ])

        # Send Telegram notification
        trade_details = {
            "timestamp": timestamp.isoformat(),
            "symbol": request.symbol,
            "transaction_type": request.transaction_type.value,
            "quantity": request.quantity,
            "price": fill_price,
            "commission": commission
        }
        message = f"""
*Paper Trade Executed*
- *Timestamp:* {trade_details['timestamp']}
- *Symbol:* {trade_details['symbol']}
- *Transaction Type:* {trade_details['transaction_type']}
- *Quantity:* {trade_details['quantity']}
- *Price:* {trade_details['price']:.2f}
- *Commission:* {trade_details['commission']:.2f}
"""
        send_telegram_message(message)

        order_id = f"paper_{uuid.uuid4()}"
        return OrderResponse(status="ok", order_id=order_id, raw={"fill_price": fill_price, "commission": commission})

    def get_quote(self, symbol: str) -> Quote:
        try:
            return self.live_broker.get_quote(symbol)
        except Exception as e:
            logging.error(f"Failed to fetch live quote for {symbol}: {e}")
            raise

    def cancel_order(self, order_id: str) -> OrderResponse:
        raise NotImplementedError("PaperBroker does not support canceling orders.")

    def modify_order(self, order_id: str, updates: Dict[str, Any]) -> OrderResponse:
        raise NotImplementedError("PaperBroker does not support modifying orders.")

    def get_orderbook(self) -> List[Dict[str, Any]]:
        return []

    def get_tradebook(self) -> List[Dict[str, Any]]:
        return []

    def get_history(self, symbol: str, interval: str, start: str, end: str) -> List[Dict[str, Any]]:
        return self.live_broker.get_history(symbol, interval, start, end)

    def download_instruments(self) -> Any:
        return self.live_broker.download_instruments()

    def get_instruments(self) -> List[Any]:
        return self.live_broker.get_instruments()

    def get_option_chain(self, underlying: str, exchange: str, **kwargs: Any) -> List[Dict[str, Any]]:
        raise NotImplementedError("PaperBroker does not support option chains.")

    def connect_websocket(self, *args, **kwargs) -> None:
        pass

    def symbols_to_subscribe(self, symbols: List[str]) -> None:
        pass

    def connect_order_websocket(self, *args, **kwargs) -> None:
        pass

    def unsubscribe(self, symbols: List[str]) -> None:
        pass

    def get_margins_required(self, orders: List[Dict[str, Any]]) -> Any:
        return 0

    def get_span_margin(self, orders: List[Dict[str, Any]]) -> Any:
        return 0

    def get_multiorder_margin(self, orders: List[Dict[str, Any]]) -> Any:
        return 0

    def get_profile(self) -> Dict[str, Any]:
        return {}

    def exit_positions(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("PaperBroker does not support exiting positions.")

    def convert_position(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("PaperBroker does not support converting positions.")
