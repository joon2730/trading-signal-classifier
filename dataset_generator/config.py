class Config:
    use_local_data = False

    if use_local_data:
        data_path = "data/raw/stock_data.csv"
    else:
        ticker = "BTC-USD"
        start_date = "2017-01-01"
        end_date = "2023-10-01"
        interval = "1d"

    target = 'Close'

    if use_local_data:
        save_path = "data/labelled/" + data_path.split('.')[0].split('/')[-1] + '.csv'
    else:
        save_path = f"data/labelled/{ticker}.csv"