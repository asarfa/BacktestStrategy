import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import yfinance as yf
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from typing import List

pd.options.mode.chained_assignment = None  # Ignore Setting With Copy Warning


class Provider(Enum):
    YAHOO = "YAHOO"


class Data:
    def __repr__(self):
        kwargs = [f"{key}={value!r}" for key, value in self.__dict__.items() if key[0] != "_" or key[:2] != "__"]
        return "{}({})".format(type(self).__name__, "".join(kwargs))


class QuoteId(Data):
    def __init__(
            self,
            product_code: str = None,
            dt: datetime = None,
            provider: Provider = None,
    ):

        if isinstance(provider, Provider):
            self.provider = provider
        else:
            raise TypeError(f"self.provider must be an instance of Provider, current type is {type(provider)}")
        self.product_code = product_code
        self.dt = dt

    def __hash__(self):
        return hash((self.product_code, self.dt, self.provider))

    def __eq__(self, other):
        return self.__dict__ == other.__dict__ if isinstance(other, self.__class__) else False


class Quote(Data):
    def __init__(
            self,
            id: QuoteId = None,
            close: float = None
    ):
        self.id = id
        self.close = close

    def __eq__(self, other):
        return self.id == other.id if isinstance(other, self.__class__) else False

    def __hash__(self):
        return hash(self.id)

    def __lt__(self, other):
        return self.id.dt < other.id.dt if isinstance(other, self.__class__) else False

    def __le__(self, other):
        return self.id.dt <= other.id.dt if isinstance(other, self.__class__) else False

    def __gt__(self, other):
        return self.id.dt > other.id.dt if isinstance(other, self.__class__) else False

    def __ge__(self, other):
        return self.id.dt >= other.id.dt if isinstance(other, self.__class__) else False

    @classmethod
    def param_json(cls, **kwargs):
        product_code = kwargs.get("Ticker")
        close = kwargs.get("Close")
        dt = kwargs.get("Date")
        id = QuoteId(product_code, dt, Provider.YAHOO)
        return cls(id=id, close=close)


class QuoteView:
    def __init__(
            self,
            product_code: str = None,
            dt: datetime = None,
            provider: Provider = None,
            close: float = None
    ):
        self.product_code = product_code
        self.dt = dt
        self.provider = provider
        self.close = close

    @classmethod
    def from_quote(cls, quote: Quote):
        return cls(product_code=quote.id.product_code, dt=quote.id.dt, provider=quote.id.provider, close=quote.close)


class Factory:

    """"
    This Class allows to transform the data into the rights Objects
    """

    @staticmethod
    def compute_quote(json: List[dict]) -> List[Quote]:
        """
        This method will convert dict of jason into Quote Object
        ------

        """
        return list(map(lambda obj: Quote.param_json(**obj), json))

    @staticmethod
    def to_quoteview(quotes: List[Quote]) -> List[QuoteView]:
        """
        This method will convert Quote Object into QuoteView Object
        ------

        """
        return list(map(lambda obj: QuoteView.from_quote(obj), quotes))


class WeightId(Data):
    def __init__(
            self,
            underlying_code: str = None,
            product_code: str = None,
            ts: datetime = None
    ):
        self.product_code = product_code
        self.underlying_code = underlying_code,
        self.ts = ts

    def __hash__(self):
        return hash((self.product_code, self.underlying_code, self.ts))

    def __eq__(self, other):
        return self.__dict__ == other.__dict__ if isinstance(other, self.__class__) else False


class Weight(Data):
    def __init__(
            self,
            id: WeightId = None,
            value: float = None
    ):
        self.id = id
        self.value = value

    def __hash__(self):
        return hash(self.id)

    def __lt__(self, other):
        return self.id.ts < other.id.ts if isinstance(other, self.__class__) else False

    def __le__(self, other):
        return self.id.ts <= other.id.ts if isinstance(other, self.__class__) else False

    def __gt__(self, other):
        return self.id.ts > other.id.ts if isinstance(other, self.__class__) else False

    def __ge__(self, other):
        return self.id.ts >= other.id.ts if isinstance(other, self.__class__) else False


class ParamsInit:
    """"
    This Class allows to get the initial parameters from the user
    These parameters will be used first for the Signal Generator and in second time for the Backtester
    """

    def __init__(
            self,
            start_ts: datetime = None,
            end_ts: datetime = None,
            underlying_ticker: List[str] = None,
            strategy_name: str = None,
            freq: int = None
    ):
        self.start_ts = start_ts
        self.end_ts = end_ts
        self.underlying_ticker = list(underlying_ticker)
        self.strategy_name = strategy_name
        self.freq = freq  # freq : Strat launch, i.e if 1 change of position every day in the ptf

    def __repr__(self):
        return str(self.__dict__)

    def __str__(self):
        return repr(self)


class Params(ParamsInit):
    """"
    This Class allows to get the final parameters in order to compute the backtest
    The final parameter is the list of Weights which would be computed from the Signal Generator
    """
    def __init__(
            self,
            weights: List[Weight] = None,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.weights = weights

    def __repr__(self):
        return super().__repr__()

    def __str__(self):
        return super().__str__()


def checkparams(func):  # been challenging
    """
    This method check if the start_date in the ParamsInit is well before the end_date
    ------

    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not args[0].params.start_ts < args[0].params.end_ts:
            raise ValueError(f"end-ts must be superior than start-ts")
        return func(*args, **kwargs)
    return wrapper


class Backtester:
    """"
    This Class executes the backtest of a strategy based on the parameters given by the users
    Several methods have been implemented in order to have a view on the levels of the strat and
    on descriptives statistics
    """
    def __init__(
            self,
            params: Params = None
    ):
        self.params = params
        self.quote_by_id = {quote.id: quote for quote in self._loader}
        self.calendar = None

    def compute_calendar(self):
        calendar = []
        temp_ts = self.params.start_ts
        while temp_ts <= self.params.end_ts:
            calendar.append(temp_ts)
            temp_ts += timedelta(days=1)
        return calendar

    def compute_levels(self):
        self.calendar = self.compute_calendar()
        if self.calendar is None:
            raise TypeError(f"self.calendar cannot be {type(None)}")
        return self._compute_levels()

    def _compute_levels(self) -> List[Quote]:
        levels = {}
        missing_days = []
        weight_by_id = {weights.id: weights for weights in self.params.weights}
        for ts in self.calendar:
            if ts == self.params.start_ts:
                levels[ts] = Quote(id=QuoteId(self.params.strategy_name, ts, Provider.YAHOO), close=100)
            else:
                temp_ptf_ret = 0
                for underlying in self.params.underlying_ticker:
                    previous_weight = weight_by_id.get(WeightId(self.params.strategy_name, underlying,
                                                                ts - timedelta(days=self.params.freq)))
                    if previous_weight is None:
                        raise TypeError(f"no weight")
                    else:
                        current_price = self.quote_by_id.get(QuoteId(underlying, ts, Provider.YAHOO))
                        if current_price is None:
                            missing_days.append([ts.isoformat(), underlying])
                            continue
                        previous_price = self.quote_by_id.get(
                            QuoteId(underlying, ts - timedelta(days=self.params.freq), Provider.YAHOO))
                        if previous_price is None:
                            continue
                        rt = current_price.close / previous_price.close - 1
                        temp_ptf_ret += previous_weight.value * rt
                previous_levels = levels.get(ts - timedelta(days=self.params.freq))
                levels[ts] = Quote(id=QuoteId(self.params.strategy_name, ts, Provider.YAHOO),
                                   close=previous_levels.close * (1 + temp_ptf_ret))
        if len(missing_days) > 0:
            print(f'----------------------------------------------------------------------'
                  f'There a missing price data on {Provider.YAHOO} for these couple(date,crypto) :'
                  f'{missing_days}'
                  f'Therefore we are assuming that the market were closed and the level '
                  f'of the ptf is the same as the day before'
                  f'----------------------------------------------------------------------')
        return list(levels.values())

    @property
    def _loader(self) -> List[Quote]:
        """
        This method will load the value needed to execute the backtest
        ------

        """
        return DataLoader(self.params, -timedelta(days=self.params.freq)).f_data_loader()
        # -timedelta(days=1) because there is a lag of one day when downloading the value wanted

    def df_levels(self) -> pd.DataFrame:
        """
        This method will convert the list of Quote object into a Dataframe
        ------

        """
        list_levels = self.compute_levels()
        return pd.DataFrame([vars(item) for item in Factory.to_quoteview(list_levels)]).set_index("dt")

    def descriptive(self):
        """
        This method will print the performance and descriptive statistics of the strategy backtested
        and return the series of the levels output
        """
        levels = self.df_levels()['close']
        perf = levels.iloc[-1] / levels.iloc[0] - 1
        perf_percent = "{:.2f}".format(perf * 100)
        r = np.log(levels / levels.shift(1)).iloc[1:-1]
        maxi = "{:.3f}".format(r.max())
        mean = "{:.3f}".format(r.mean())
        mini = "{:.3f}".format(r.min())
        standard = "{:.3f}".format(r.std())
        kurto = "{:.3f}".format(r.kurtosis())
        skewn = "{:.3f}".format(r.skew())
        print(f'Backtest Results of {self.params.strategy_name} Strategy : ')
        print(f'From {self.params.start_ts.date()} to {self.params.end_ts.date()}')
        print(f'--------------------------------------------------')
        print(f'The performance is : {perf_percent} %')
        print(f'--------------------------------------------------')
        print(f'Descriptive Statistics of the Stragety"s return')
        print(f'--------------------------------------------------')
        print(f'max     min     mean     std     kurto    skew')
        print(f'{maxi}   {mini}   {mean}   {standard}   {kurto}   {skewn}')
        print(f'--------------------------------------------------')

        return levels


class DataLoader:
    """"
    This Class download the data required from yahoo finance, do some transformation of the data to get in fit with
    the method of the class Factory and convert it into object Quote
    """

    def __init__(
            self,
            params: ParamsInit = None,
            delta: timedelta = None
    ):
        """
        Parameters
        ----------
        params : the initial parameters the user gives
        delta : a numbers of day before the start date of the backtest to get the data needed in order to generate
        signal and compute the strategy at the strat_date
        """

        self.params = params
        self.delta = delta

    def compute(self) -> pd.DataFrame:
        """
        This method will compute a dataframe of crypto's ptf values from yahoo finance api
        ------

        """
        return yf.download(" ".join(self.params.underlying_ticker), start=self.params.start_ts - self.delta,
                           end=self.params.end_ts + timedelta(self.params.freq), group_by='tickers')

    @staticmethod
    def transform(df: pd.DataFrame, ticker: str) -> dict:
        """
        This method will add two columns to the dataframe to get in fit with the method from class Factory
        pd.options.mode.chained_assignment = None as been set due to the message warning exiting from this function
        ------

        """
        tc = df[ticker]
        tc["Ticker"] = ticker
        tc["Date"] = tc.index
        return tc.to_dict(orient="index")

    def f_data_loader(self) -> List[Quote]:
        """
        This method will convert the dataframe into Quote object
        Not been able to do without a loop (it is the only one of the project)
        ------

        """
        df = self.compute()
        json_format = []
        list_quote = []
        for ticker in self.params.underlying_ticker:
            json_format.extend(value for key, value in self.transform(df, ticker).items())
            list_quote.extend(Factory.compute_quote(json_format))
        return list_quote


class AsymmetricMeasure:
    """"
    This Class computed an asymmetric measure of a crypto-currency from a start date to an end date
    The asymmetric measure computed refers to the one mentioned on the research paper
    Called the asymmetric response measure of Baur and Dimpfl
    The measure is calculated based on quantile autoregressive model
    """

    def __init__(
            self,
            ticker: str = None,
            df: pd.DataFrame = None,
            start_date: datetime = None,
            end_date: datetime = None
    ):
        """
        Parameters
        ----------
        ticker : a ticker of the list of the portfolio
        df : a dataframe in which there is all the information of the portfolio's asset's prices
        start_date : a datetime for the debut of the period
        end_date : a datetime for the end of the period to compute the measure
        """

        self.ticker = ticker
        self.df = df
        self.start_date = start_date
        self.end_date = end_date
        self.quantiles_down = np.arange(0.01, 0.11, 0.01)
        self.quantiles_up = np.flip(1 - self.quantiles_down)
        self.model = smf.quantreg("r ~ rshift", self.load_returns())

    def load_returns(self) -> pd.DataFrame:
        """
        This method will compute a dataframe based on the information provided in parameters
        This dataframe will be used in the estimation of the quantile autoregressive model
        ------

        """
        tick = self.df[self.ticker]["Close"].loc[self.start_date:self.end_date]
        returns = np.log(tick / tick.shift(1))
        r = returns.iloc[2:].reset_index(drop=True)
        rshift = returns.iloc[1:-1].reset_index(drop=True)
        returns_qar = pd.concat([r, rshift], axis=1)
        returns_qar.columns = ["r", "rshift"]
        return returns_qar

    def fit_model(self, q: float, sum_fit: float) -> float:
        """
        This method is based on a model coming from the package statsmodels
        It estimates a quantile regression model using iterative reweighted least squares
        This method will compute the autoregressive parameter for a quantile given
        ------

        """
        res = self.model.fit(q=q, max_iter=10000)
        return sum_fit + res.params["rshift"]

    def loop_model(self, quantile_list: np.ndarray) -> float:
        """
        This method will return the sum of the autoregressive parameter for a list of quantile given
        Divided by the number of quantile in the list
        There is the lower quantile list and the upper quantile list
        With the aim to obtain robust estimate for extreme quantiles
        In particular for shorter period to compute the measure
        ------

        """
        sum_fit = 0
        sum_fit = sum(self.fit_model(q, sum_fit) for q in quantile_list)
        return sum_fit / len(quantile_list)

    def compute(self) -> float:
        """
        This method will compute the asymmetric response measure of Baur and Dimpfl
        ------

        """
        return self.loop_model(self.quantiles_down) - self.loop_model(self.quantiles_up)


class RollingAsymmetricMeasure:
    """
    This Class will compute at each Params.freq the rolling (timedelta to choose) asymmetric measure
    of crypto's portfolio for all backtest period
    """

    def __init__(
            self,
            params: ParamsInit = None,
            delta: timedelta = None
    ):
        """
        Parameters
        ----------
        params : the initial parameters the user gives
        delta : a numbers of day in which the rolling asymmetric measure will be computed
        """

        self.params = params
        self.delta = delta
        self.df = DataLoader(self.params, self.delta).compute()

    def dict_asymm(self, start: datetime, end: datetime) -> dict:
        """
        This method will compute a dictionary of asymmetric measure for all crypto's portfolio
        From a start to an end date based on the Class created above (AsymmetricMeasure)
        Therefore each key (ticker) will receive one value (the asymmetric measure)

        ------

        """
        return dict(map(lambda t: (t, AsymmetricMeasure(t, self.df, start, end).compute()),
                        self.params.underlying_ticker))

    def compute(self) -> dict:
        """
        This method will compute a nested dictionary of rolling asymmetric measure for all crypto's portfolio
        And for all the backtest period
        Therefore each key (end date of the rolling period) will receive one dict composed of two value :
        the ticker (key) and the asymmetric measure (value)

        ------

        """
        start = self.df.index[0]
        end = start + self.delta
        stop = self.df.index[-1]
        dict_asym_meas = {end: self.dict_asymm(start, end)}
        while end < stop:
            start = start + timedelta(days=self.params.freq)
            end = end + timedelta(days=self.params.freq)
            dict_asym_meas[end] = self.dict_asymm(start, end)
        return dict_asym_meas


class SignalGenerator:
    """
    This Class will compute the list of Weights according to a strategy based on the paper disclosure
    The strategy is implemented through the rolling asymmetric measure calculated on the Class created above
    """
    def __init__(
            self,
            params: ParamsInit = None,
            delta: timedelta = None
    ):
        """
        Parameters
        ----------
        Same as the Class above without the dataframe of crypto's value
        """

        self.params = params
        self.delta = delta

    @checkparams
    def compute(self) -> List[Weight]:
        """
        This method will compute a list of Weights for all crypto's and for all the backtest period (given a frequence)
        Based on the strategy (quiet simple) defined by the function below
        For each date signal the method compute the number of positive asymmetric measure and get the value of the
        asymmetric measure for each crypto in the dict associated
        Then called the function strat to compute the weight

        ------

        """
        dict_asym_meas = RollingAsymmetricMeasure(self.params, self.delta).compute()
        start = self.params.start_ts
        weights = []
        while start < self.params.end_ts:
            date_signal = start - timedelta(days=self.params.freq)
            nb_pos = sum(value < 0 for value in dict_asym_meas.get(date_signal).values())
            weights.extend(list(map(lambda ticker: Weight(id=WeightId(self.params.strategy_name, ticker, start),
                                                          value=self.strat(dict_asym_meas.get(date_signal).get(ticker),
                                                          nb_pos)), self.params.underlying_ticker)))
            start = start + timedelta(days=self.params.freq)
        return weights

    @staticmethod
    def strat(nb_to_test: float, nb_pos: int) -> float:
        """
        This method will compute a weight for each crypto at each date
        If the asymmetric measure of a crypto at this date is positive then its weight is 0
        If it is negative its weight is 1/total of positive asymmetric measure
        Indeed the paper emphasis : positive shocks increase the volatility by more than negative shocks
        Therefore the weights are either 0 or 1/(nb_positive_asymmetric_measure)
        The strat could have been more technical : the more negative asymmetric measure the more the crypto's weight
        in the ptf

        ------

        """
        return 1 / nb_pos if nb_to_test < 0 else 0

    @checkparams
    def compute_equal_weighted(self) -> List[Weight]:
        """
        This method will compute a list of Weights for all crypto's and for all the backtest period (given a frequence)
        The weights will be equally weighted at every date so it isn't a truly strategy
        This method has been implemented in order to compare the performance with the strategy above

        ------

        """
        true_start = self.params.start_ts
        weights = []
        while true_start < self.params.end_ts:
            weights.extend(list(map(lambda tickers: Weight(id=WeightId(self.params.strategy_name, tickers, true_start),
                                                           value=1 / len(self.params.underlying_ticker)),
                                    self.params.underlying_ticker)))
            true_start = true_start + timedelta(days=self.params.freq)
        return weights


if __name__ == '__main__':

    ptf = ["BTC-USD", "ETH-USD", "ADA-USD", "DOGE-USD", "SOL1-USD", "SHIB-USD", "XRP-USD"]
    start = datetime(2021, 5, 30)
    end = datetime(2021, 11, 27)

    parinit = ParamsInit(start, end, ptf, "High_Vol", 1)
    weights = SignalGenerator(parinit, timedelta(days=7)).compute()
    par = Params(weights, start_ts=start, end_ts=end, underlying_ticker=ptf, strategy_name="High_Vol", freq=1)
    dataframe_levels = Backtester(par).descriptive()

    parinit_ew = ParamsInit(start, end, ptf, "Equally_Weighted", 1)
    weights_ew = SignalGenerator(params=parinit_ew, delta=timedelta(days=7)).compute_equal_weighted()
    par_ew = Params(weights_ew, start_ts=start, end_ts=end,
                    underlying_ticker=ptf, strategy_name="Equally_Weighted", freq=1)
    dataframe_levels_ew = Backtester(par_ew).descriptive()
