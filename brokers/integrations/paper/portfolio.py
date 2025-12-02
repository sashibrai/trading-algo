from __future__ import annotations
from typing import Dict
from ...core.schemas import Position, Quote, OrderRequest
from ...core.enums import TransactionType

class Portfolio:
    """
    Manages the virtual portfolio for paper trading.
    """

    def __init__(self, initial_cash: float = 100000.0):
        self.cash = initial_cash
        self.positions: Dict[str, Position] = {}
        self.realized_pnl = 0.0

    def update_position(self, request: OrderRequest, fill_price: float, commission: float):
        """
        Updates the portfolio with a new trade.
        """
        if request.transaction_type == TransactionType.BUY:
            self.cash -= (fill_price * request.quantity) + commission
        else:
            self.cash += (fill_price * request.quantity) - commission

        if request.symbol in self.positions and self.positions[request.symbol].quantity_total != 0:
            position = self.positions[request.symbol]

            current_quantity = position.quantity_total
            trade_quantity = request.quantity if request.transaction_type == TransactionType.BUY else -request.quantity
            new_quantity = current_quantity + trade_quantity

            # If signs are different or position is closed, it's a closing trade (partially or fully)
            if current_quantity * new_quantity <= 0:
                closed_quantity = min(abs(current_quantity), abs(trade_quantity))

                if current_quantity > 0: # Closing a long position
                    profit = (fill_price - position.average_price) * closed_quantity
                else: # Closing a short position
                    profit = (position.average_price - fill_price) * closed_quantity

                self.realized_pnl += profit

            # If signs are the same, it's an opening/increasing trade. Update average price.
            if new_quantity != 0 and (current_quantity * new_quantity > 0):
                position.average_price = ((position.average_price * current_quantity) + (fill_price * trade_quantity)) / new_quantity
            # If we are flipping the position, the average price of the new position is the fill_price of this trade
            elif new_quantity != 0:
                position.average_price = fill_price

            position.quantity_total = new_quantity
            # If position is closed, remove it
            if new_quantity == 0:
                del self.positions[request.symbol]

        else: # New position
            position_quantity = request.quantity if request.transaction_type == TransactionType.BUY else -request.quantity
            self.positions[request.symbol] = Position(
                symbol=request.symbol,
                exchange=request.exchange,
                quantity_total=position_quantity,
                quantity_available=position_quantity,
                average_price=fill_price,
                pnl=0,
                product_type=request.product_type,
                raw={}
            )

    def get_unrealized_pnl(self, quotes: Dict[str, Quote]) -> float:
        """
        Calculates the unrealized P&L for all positions.
        """
        unrealized_pnl = 0.0
        for symbol, position in self.positions.items():
            if symbol in quotes:
                unrealized_pnl += (quotes[symbol].last_price - position.average_price) * position.quantity_total
        return unrealized_pnl

    def get_total_pnl(self, quotes: Dict[str, Quote]) -> float:
        """
        Calculates the total P&L (realized + unrealized).
        """
        return self.realized_pnl + self.get_unrealized_pnl(quotes)
