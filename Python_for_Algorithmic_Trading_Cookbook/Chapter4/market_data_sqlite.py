# libraries 
from sys import argv
import sqlite3
import pandas as pd 
import exchange_calendars as xcals
from openbb import obb
obb.user.preferences.output_type = "dataframe"

# get_stock function from the previous recipe 
def get_stock_data(symbol, start_date=None, end_date=None):
    data = obb.equity.price.historical(
    symbol,
    start_date = start_date,
    end_date = end_date,
    provider="yfinance"
    )
    data.reset_index(inplace=True)
    data['symbol'] = symbol
    return data


# we need to modify save_data_range func to use pandas to_sql method
def save_data_range(symbol, conn, start_date,
   end_date):
    data = get_stock_data(symbol, start_date, end_date)
    data.to_sql(
        "stock_data",
        conn,
        if_exists="replace",
        index=False
)

# create a func that grabs data from the last trading day based on exchange calendar
def save_last_trading_session(symbol, conn, today):
    data = get_stock_data(symbol, today, today)
    data.to_sql(
        "stock_data",
        conn,
        if_exists="append",
        index=False
    )

# script’s main execution code
if __name__ == "__main__":
    conn = sqlite3.connect("market_data.sqlite")
    if argv[1] == "bulk":
        symbol = argv[2]
        start_date = argv[3]
        start_date = argv[4]
        save_data_range(symbol, conn, start_date=None, end_date=None)
        print(f"{symbol} saved between {start_date} and {start_date}")
    elif argv[1] == "last":
        symbol = argv[2]
        calendar = argv[3]
        cal = xcals.get_calendar(calendar)
        today = pd.Timestamp.today().date()
        if cal.is_session(today):
            save_last_trading_session(symbol, conn, today)
            print(f"{symbol} saved")
        else:
            print(f"{today} is not a trading day. Doing nothing!")
    else:
        print("Enter bulk or last")