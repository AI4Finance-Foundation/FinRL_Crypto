# FinRL_Crypto: Address Overfitting DRL Agents for Cryptocurrency Trading  

![Banner](https://user-images.githubusercontent.com/69801109/214294114-a718d378-6857-4182-9331-20869d64d3d9.png)  

For financial reinforcement learning (FinRL), we provide a way to address the dreaded overfitting trap and increase your chances of success in the wild world of cryptocurrency trading. Our approach has been tested on 10 different currencies and during a market crash period, and has proven to be more profitable than the competition. So, don't just sit there, join us on our journey to the top of the crypto mountain!  

## Paper  
Our [paper](https://arxiv.org/abs/2209.05559)  

## How to Use FinRL_Crypto with Binance API  

This repository is ready for work with Binance API for cryptocurrency trading using Deep Reinforcement Learning (DRL).  

### 📋 What's Already Done  
✅ **Updated Dependencies** - All packages are compatible with Python 3.8+ (and recommended for Python 3.10).  
✅ **Fixed Deprecated Methods** - Code is compatible with new versions of pandas and numpy.  
✅ **Added Error Handling** - Improved stability during execution.  
✅ **Optimized Performance** - Replaced outdated pandas methods with more efficient alternatives.  

### 🔧 Requirements  
- **Python 3.10** (Recommended version)  
- Binance API keys  

### 🎯 Setup  

#### 1. Install Dependencies  
**For Python 3.10 (Recommended):**  
```bash
pip install -r requirements-python310.txt
```

**For Python 3.11+:**  
```bash
pip install -r requirements.txt
```

#### 2. Configure Binance API Keys  

1. Log in to your Binance account → **API Management**  
2. Create new API keys  
3. Open `config_api.py` and replace placeholders:  

```python
API_KEY_BINANCE = 'YOUR_PUBLIC_API_KEY'
API_SECRET_BINANCE = 'YOUR_SECRET_API_KEY'
```

#### 3. Verify Configuration  

Main settings are configured in `config_main.py`:  
- **TIMEFRAME**: `'5m'` - 5-minute candlesticks  
- **TICKER_LIST**: List of cryptocurrencies for trading (e.g., `['BTC', 'ETH']`))  
- **TECHNICAL_INDICATORS**: Technical indicators (MACD, RSI, CCI, DX)  

## 🎯 Quick Start Guide  

### 1. Download Data  
```bash
python 0_dl_trainval_data.py  # Download training/validation data
python 0_dl_trade_data.py     # Download trade data
```  

### 2. Optimize Hyperparameters  
Choose one of the following optimization schemes:  
- **Combinatorial Purged Cross-validation**:  
  ```bash
  python 1_optimize_cpcv.py 
  ```
- **K-Fold Cross-validation**:  
  ```bash
  python 1_optimize_kcv.py 
  ```
- **Walk-forward validation**:  
  ```bash
  python 1_optimize_wf.py 
  ```

### 3. Validate Trained Agents  
```bash
python 2_validate.py 
```

### 4. Backtest Trained Agents  
```bash
python 4_backtest.py 
```

## 🔍 Core Components  

### 📁 Project Files  
- **`config_main.py`**: Main configuration file.  
- **`config_api.py`**: Binance API keys.  
- **`processor_Binance.py`**: Data processing for Binance API.  
- **`environment_Alpaca.py`** (Note: Alpaca is a separate trading environment; this project focuses on Binance).  
- **`drl_agents/`**: DRL algorithms (e.g., PPO, A2C, DDPG, TD3, SAC).  

### 💡 Supported Cryptocurrencies  
- AAVE, AVAX, BTC, ETH, LINK, LTC, MATIC, NEAR, SOL, UNI  
- New cryptocurrencies can be added to `config_main.py`.  

### 📊 Technical Indicators  
- **MACD (Moving Average Convergence Divergence)**  
- **RSI (Relative Strength Index)**  
- **CCI (Commodity Channel Index)**  
- **DX (Directional Movement Index)**  

## ⚠️ Important Notes  

1. **API Keys**: Never store API keys in public repositories.  
2. **Testing**: Test on a demo account first.  
3. **Risks**: Cryptocurrency trading involves high risks.  
4. **Versions**: Ensure you are using Python 3.8+.  

## 🔧 Troubleshooting  

### Error: "API keys not configured"  
- Check `config_api.py`  
- Ensure API keys are entered correctly  

### Error: "No data returned"  
- Check internet connectivity  
- Ensure cryptocurrency symbols are correct  
- Check Binance API rate limits  

### Dependency Issues  
```bash
# Install TA-Lib (if needed)
conda install -c conda-forge ta-lib  # For Anaconda
# or for macOS:
brew install ta-lib 
```  

## 📈 Additional Features  

### Visualization of Results  
Results are saved in the `/plots_and_metrics/` folder.  

### Customizing Parameters  
- Modify dates in `config_main.py`  
- Tune technical indicators (e.g., MACD parameters)  
- Adjust cryptocurrency lists in `config_main.py`  

## 🆘 Support  

If you encounter issues:  
1. Check error logs for clues.  
2. Verify all dependencies are installed (run `pip install -r requirements.txt` or `pip install -r requirements-python310.txt`).  
3. Confirm Binance API keys are correctly configured in `config_api.py`.  

---

**Ready to Go! 🚀**  
This project is fully configured for cryptocurrency trading on Binance using advanced Deep Reinforcement Learning algorithms.