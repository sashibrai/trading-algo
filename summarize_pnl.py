import csv
from datetime import datetime

def generate_pnl_summary():
    """
    Generates a P&L summary from the trade and P&L logs.
    """
    print("--- End-of-Day P&L Summary ---")

    try:
        with open("paper_trades.csv", "r") as f:
            trades = list(csv.reader(f))
            num_trades = len(trades) - 1
            total_commission = sum(float(trade[5]) for trade in trades[1:])
    except FileNotFoundError:
        print("No trades were made today.")
        num_trades = 0
        total_commission = 0

    try:
        with open("pnl_log.csv", "r") as f:
            pnl_entries = list(csv.reader(f))
            if len(pnl_entries) > 1:
                last_pnl_entry = pnl_entries[-1]
                realized_pnl = float(last_pnl_entry[1])
                unrealized_pnl = float(last_pnl_entry[2])
                total_pnl = float(last_pnl_entry[3])
            else:
                realized_pnl = 0
                unrealized_pnl = 0
                total_pnl = 0
    except FileNotFoundError:
        realized_pnl = 0
        unrealized_pnl = 0
        total_pnl = 0

    print(f"Date: {datetime.now().strftime('%Y-%m-%d')}")
    print(f"Number of trades: {num_trades}")
    print(f"Total commission: {total_commission:.2f}")
    print(f"Realized P&L: {realized_pnl:.2f}")
    print(f"Unrealized P&L: {unrealized_pnl:.2f}")
    print(f"Total P&L: {total_pnl:.2f}")

    print("--- End of Summary ---")

if __name__ == "__main__":
    generate_pnl_summary()
