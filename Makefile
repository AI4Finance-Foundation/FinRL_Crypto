.PHONY: all 0_dl_trade_data.py 0_dl_trainval_data.py 1_optimize_cpcv.py 1_optimize_kcv.py 1_optimize_wf.py

all: 0_dl_trade_data.py 0_dl_trainval_data.py 1_optimize_cpcv.py 1_optimize_kcv.py 1_optimize_wf.py

01_dl_trainval_data.py:
	./.venv/bin/python 0_dl_trainval_data.py

02_dl_trade_data.py:
	./.venv/bin/python 0_dl_trade_data.py

11_optimize_cpcv.py:
	./.venv/bin/python 1_optimize_cpcv.py

12_optimize_kcv.py:
	./.venv/bin/python 1_optimize_kcv.py

13_optimize_wf.py:
	./.venv/bin/python 1_optimize_wf.py

21_validate.py:
	./.venv/bin/python 2_validate.py

41_backtest.py:
	./.venv/bin/python 4_backtest.py


# TODO: papper trade
# TODO: live trade