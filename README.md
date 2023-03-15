# FinRL_Crypto: Address Overfitting Your DRL Agents for Cryptocurrency Trading 

![banner](https://user-images.githubusercontent.com/69801109/214294114-a718d378-6857-4182-9331-20869d64d3d9.png)

For financial reinforcement learning (FinRL), we've found a way to address the dreaded overfitting trap and increase your chances of success in the wild world of cryptocurrency trading. Our approach has been tested on 10 different currencies and during a market crash period, and has proven to be more profitable than the competition. So, don't just sit there, join us on our journey to the top of the crypto mountain! 

## Paper

The original [paper](https://arxiv.org/abs/2209.05559) 

## How to use

To reproduce the results in the paper, the codes are simplified as much as possible. You start with the settings in```config_main.py``` file, where you set all the settings for:

- The Walkforward, K-Cross Validation, and Combinatorial Purged Cross Validation (CPCV) methods.
- Set how many candles/data points you require for training and validation.
- Set which tickers you will download from Binance, the minimum buy limits.
- Set your technical indicators.
- Computes automatically the exact start and end dates for training and validation, respectively, based on your trade start date and end date.

A short description of each folder:
- ```data``` Contains all your training/validation data in the main folder, and a subfolder which contains ```trade_data``` after download using both ```0_dl_trainval_data.py``` and ```0_dl_trade_data.py``` (more later)
- ```drl_agents``` Contains the DRL framework [ElegantRL]([/guides/content/editing-an-existing-page](https://arxiv.org/abs/2209.05559)) which implements a series of model-free DRL algorithms
- ```plots_and_metrics``` Dump folder for all analysis images and performance metrics produced
- ```train``` Holds all utility functions for DRL training
- ```train_results``` After running either ```1_optimize_cpcv.py``` /  ```1_optimize_kcv.py``` / ```1_optimize_wf.py``` will have a folder with your trained DRL agents

Then, running and producing similar results to that in the paper are simple, following the numbered Python files as indicated by the number of the filename:

- ```0_dl_trainval_data.py```  Downloads the train and validation data according to ```config_main.py```
- ```0_dl_trade_data.py``` Downloads the trade data according to ```config_main.py```
- ```1_optimize_cpcv.py``` Optimizes hyperparameters with a Combinatorial Purged Cross-validation scheme
- ```1_optimize_kcv.py``` Optimizes hyperparameters with a K-Fold Cross-validation scheme
- ```1_optimize_wf.py``` Optimizes hyperparameters with a Walk-forward validation scheme
- ```2_validate.py``` Shows insights about the training and validation process (select a results folder from train_results)
- ```4_backtestpy``` Backtests trained DRL agents (enter multiple results folders from train_results in a list)
- ```5_pbo.py``` Computes PBO for trained DRL agents (enter multiple results folders from train_results in a list)

Simply run the scripts in the above order. Please note the trained agents are auto-saved to the folder ```train_results```. That is where you can find your trained DRL agents!

## Citing FinRL_Crypto

```
@article{gort2022deep,
  title={Deep reinforcement learning for cryptocurrency trading: Practical approach to address backtest overfitting},
  author={Gort, Berend Jelmer Dirk and Liu, Xiao-Yang and Gao, Jiechao and Chen, Shuaiyu and Wang, Christina Dan},
  journal={AAAI Bridge on AI for Financial Services},
  year={2023}
}
```
