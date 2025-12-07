# CCXT Environment Integration Guide

## Overview
This guide explains how to use the new `CryptoEnvCCXT` class that replaces `CryptoEnvAlpaca` for cryptocurrency trading using the CCXT library.

## Key Changes

### 1. New Environment Class
- **File**: `environment_CCXT.py`
- **Class**: `CryptoEnvCCXT`
- **Supports**: Real crypto trading via CCXT library

### 2. Updated Configuration
- **File**: `config_main.py`
- **New sections**:
  - `CRYPTO_LIMITS`: Minimum order sizes for cryptocurrencies
  - `CCXT_CONFIG`: Exchange configuration
  - `ENV_PARAMS_CCXT`: Environment parameters
  - `TRADING_PARAMS`: Trading-specific settings

### 3. Updated Scripts
All training scripts now use `CryptoEnvCCXT` by default:
- `4_backtest.py`
- `1_optimize_cpcv.py`
- `1_optimize_kcv.py`
- `1_optimize_wf.py`

## Usage

### Basic Usage

```python
from environment_CCXT import CryptoEnvCCXT
from config_main import ENV_PARAMS_CCXT, TRADING_PARAMS

# Initialize environment
env = CryptoEnvCCXT(
    config=config,  # From processor_Binance.run()
    env_params=ENV_PARAMS_CCXT,
    initial_capital=TRADING_PARAMS['initial_capital'],
    exchange_name='binance'  # or any CCXT-supported exchange
)

# Reset and use environment
state = env.reset()
action = model.predict(state)
next_state, reward, done, info = env.step(action)
```

### Data Processing

```python
from processor_Binance import BinanceProcessor

# Initialize processor
processor = BinanceProcessor()

# Get data (now returns config dict for CCXT compatibility)
data, price_array, tech_array, time_array, config = processor.run(
    ticker_list=TICKER_LIST,
    start_date=start_date,
    end_date=end_date,
    time_interval='5m',
    technical_indicator_list=TECHNICAL_INDICATORS_LIST,
    if_vix=False
)
```

## Key Features

### 1. Exchange Support
- **Default**: Binance
- **Supported**: Any CCXT-supported exchange (100+ exchanges)
- **Configuration**: Via `config_api.py` or environment variables

### 2. Crypto-Specific Adaptations
- **24/7 Trading**: Cooldown periods adjusted for crypto markets
- **Commission Structure**: 0.1% fees (Binance standard)
- **Order Sizes**: Crypto-specific minimum amounts
- **Safety Margins**: 5% safety factor for orders

### 3. Backward Compatibility
- All existing training scripts work unchanged
- Same API as `CryptoEnvAlpaca`
- Compatible with existing data processing pipeline

## Configuration

### Environment Parameters
```python
ENV_PARAMS_CCXT = {
    'lookback': 50,            # Lookback window
    'norm_cash': 1e-6,         # Cash normalization
    'norm_stocks': 100,        # Crypto position normalization
    'norm_tech': 1,            # Technical indicator normalization
    'norm_reward': 1,          # Reward normalization
    'norm_action': 1,          # Action normalization
}
```

### Trading Parameters
```python
TRADING_PARAMS = {
    'initial_capital': 1000000,    # $1M starting capital
    'buy_cost_pct': 0.001,         # 0.1% buy fee
    'sell_cost_pct': 0.001,        # 0.1% sell fee
    'gamma': 0.99,                # Discount factor
    'safety_factor': 0.95,        # 5% safety margin
    'cooldown_periods': 24,        # 2 hours for 5m timeframe
    'forced_sell_pct': 0.05,       # 5% forced sell after cooldown
}
```

### Exchange Configuration
```python
CCXT_CONFIG = {
    'exchange_name': 'binance',  # Exchange name
    'sandbox': False,           # Set to True for testing
    'enable_rate_limit': True,
    'timeout': 30000,          # 30 seconds
    'verbose': False,
}
```

## API Keys Setup

Update `config_api.py` with your exchange API keys:

```python
API_KEY_BINANCE = 'your_binance_api_key'
API_SECRET_BINANCE = 'your_binance_secret_key'
```

## Testing

Run the test suite to verify installation:

```bash
# Test basic environment functionality
python test_ccxt_environment.py

# Test with synthetic data
python test_env_logic.py

# Test with real data (requires API keys)
python test_with_real_data.py
```

## Migration from Alpaca

### Automatic Migration
All training scripts automatically use `CryptoEnvCCXT` - no code changes needed.

### Manual Migration
If you need to use the old environment:
```python
# Old way (still available)
from environment_Alpaca import CryptoEnvAlpaca

# New way (recommended)
from environment_CCXT import CryptoEnvCCXT
```

## Live Trading Considerations

### Safety Features
- **Sandbox Mode**: Test with `sandbox=True` first
- **Order Size Limits**: Enforced by `CRYPTO_LIMITS`
- **Safety Factors**: 5-10% margins on all calculations
- **Cooldown Periods**: Prevents overtrading

### Risk Management
- Start with small capital amounts
- Use sandbox mode for testing
- Monitor API rate limits
- Set appropriate stop-losses

## Supported Exchanges

The CCXT environment supports 100+ exchanges including:
- Binance
- Coinbase Pro
- Kraken
- KuCoin
- Bybit
- BitMEX
- And many more...

Check the [CCXT documentation](https://github.com/ccxt/ccxt) for the full list.

## Troubleshooting

### Common Issues

1. **API Key Errors**: Update `config_api.py` with valid keys
2. **Data Download Failures**: Check internet connection and rate limits
3. **Order Size Errors**: Ensure `CRYPTO_LIMITS` match exchange requirements
4. **Memory Issues**: Reduce `lookback` parameter for large datasets

### Debug Mode
Enable verbose logging:
```python
env = CryptoEnvCCXT(
    ...,
    if_log=True  # Enable detailed logging
)
```

## Next Steps

1. **Backtesting**: Run existing training scripts with historical data
2. **Paper Trading**: Test with sandbox mode
3. **Live Trading**: Start with small capital amounts
4. **Optimization**: Tune parameters for your specific strategy

## Support

For issues or questions:
1. Check the test files for usage examples
2. Review CCXT documentation for exchange-specific settings
3. Verify API keys and exchange permissions