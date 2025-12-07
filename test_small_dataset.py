#!/usr/bin/env python3
"""
Test CCXT environment with small datasets.
"""

import numpy as np
from datetime import datetime, timedelta

from config_main import TICKER_LIST, TECHNICAL_INDICATORS_LIST
from processor_Binance import BinanceProcessor
from environment_CCXT import CryptoEnvCCXT


def test_with_small_dataset():
    """Test environment with very small dataset."""
    print("🧪 Testing CCXT Environment with Small Dataset")
    print("=" * 50)

    try:
        # Initialize processor
        processor = BinanceProcessor()

        # Very small dataset - 2 hours of 15m data
        end_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        start_date = (datetime.now() - timedelta(hours=2)).strftime("%Y-%m-%d %H:%M:%S")

        print(f"📅 Testing period: {start_date} to {end_date}")

        # Download data
        data, price_array, tech_array, time_array, config = processor.run(
            ticker_list=TICKER_LIST[:2],  # Use only 2 tickers
            start_date=start_date,
            end_date=end_date,
            time_interval='15m',
            technical_indicator_list=TECHNICAL_INDICATORS_LIST,
            if_vix=False
        )

        print(f"✓ Data loaded: Price {price_array.shape}, Tech {tech_array.shape}")

        # Create custom environment parameters for small dataset
        small_env_params = {
            'lookback': min(10, price_array.shape[0] - 2),  # Very small lookback
            'norm_cash': 1e-6,
            'norm_stocks': 100,
            'norm_tech': 1,
            'norm_reward': 1,
            'norm_action': 1,
        }

        # Initialize environment
        env = CryptoEnvCCXT(
            config=config,
            env_params=small_env_params,
            initial_capital=10000,
            exchange_name='binance'
        )

        print(f"✓ Environment initialized with lookback={small_env_params['lookback']}")

        # Test reset
        state = env.reset()
        print(f"✓ Environment reset: State shape {state.shape}")

        # Test a few steps
        for step in range(min(5, env.max_step)):
            action = np.random.uniform(-0.001, 0.001, env.action_dim)
            state, reward, done, info = env.step(action)
            print(f"  Step {step}: Reward {reward:.6f}, Portfolio ${env.get_portfolio_value():,.2f}")

            if done:
                print(f"  Environment done after {step + 1} steps")
                break

        env.close()
        print("✓ Small dataset test passed!")
        return True

    except Exception as e:
        print(f"❌ Small dataset test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_with_small_dataset()
    print(f"\n{'🎉 Success!' if success else '❌ Failed!'}")
    exit(0 if success else 1)