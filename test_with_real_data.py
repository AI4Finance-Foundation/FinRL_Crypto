#!/usr/bin/env python3
"""
Test script for CCXT environment with real Binance data.
This script tests the environment with actual market data.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from config_main import TICKER_LIST, TECHNICAL_INDICATORS_LIST, ENV_PARAMS_CCXT, TRADING_PARAMS
from processor_Binance import BinanceProcessor
from environment_CCXT import CryptoEnvCCXT


def test_with_real_binance_data():
    """Test CCXT environment with real data from Binance."""
    print("🔥 Testing CCXT Environment with Real Binance Data")
    print("=" * 60)

    try:
        # Initialize Binance processor
        print("📡 Connecting to Binance...")
        processor = BinanceProcessor()

        # Set time range for testing (last 24 hours for quick test)
        end_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        start_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")

        print(f"📅 Testing period: {start_date} to {end_date}")
        print(f"💱 Tickers: {TICKER_LIST}")

        # Download and process data
        print("⬇️  Downloading market data...")
        data, price_array, tech_array, time_array, config = processor.run(
            ticker_list=TICKER_LIST,
            start_date=start_date,
            end_date=end_date,
            time_interval='5m',
            technical_indicator_list=TECHNICAL_INDICATORS_LIST,
            if_vix=False
        )

        print(f"✓ Data loaded successfully!")
        print(f"  - Price array shape: {price_array.shape}")
        print(f"  - Tech array shape: {tech_array.shape}")
        print(f"  - Time steps: {len(time_array)}")

        # Initialize CCXT environment
        print("\n🤖 Initializing CCXT Environment...")
        env = CryptoEnvCCXT(
            config=config,
            env_params=ENV_PARAMS_CCXT,
            initial_capital=100000,  # Smaller capital for testing
            buy_cost_pct=TRADING_PARAMS['buy_cost_pct'],
            sell_cost_pct=TRADING_PARAMS['sell_cost_pct'],
            gamma=TRADING_PARAMS['gamma'],
            exchange_name='binance'
        )

        print(f"✓ Environment initialized!")
        print(f"  - Action dimension: {env.action_dim}")
        print(f"  - State dimension: {env.state_dim}")
        print(f"  - Max steps: {env.max_step}")

        # Test environment with real data
        print("\n🎮 Testing environment with real data...")
        state = env.reset()
        print(f"✓ Environment reset successfully")
        print(f"  - Initial portfolio: ${env.get_portfolio_value():,.2f}")

        # Run a few steps
        portfolio_values = []
        for step in range(20):  # Test 20 steps
            # Simple random action
            action = np.random.uniform(-0.01, 0.01, env.action_dim)
            state, reward, done, info = env.step(action)

            portfolio_value = env.get_portfolio_value()
            portfolio_values.append(portfolio_value)

            if step % 5 == 0:
                positions = env.get_positions()
                print(f"  Step {step}: Portfolio ${portfolio_value:,.2f}, Positions: {len(positions)} active")

            if done:
                print(f"  Environment finished after {step + 1} steps")
                break

        # Performance summary
        print("\n📊 Performance Summary:")
        print(f"  - Initial portfolio: ${portfolio_values[0]:,.2f}")
        print(f"  - Final portfolio: ${portfolio_values[-1]:,.2f}")
        print(f"  - Total return: {(portfolio_values[-1] / portfolio_values[0] - 1) * 100:.2f}%")
        print(f"  - Volatility: {np.std(portfolio_values) / np.mean(portfolio_values) * 100:.2f}%")

        # Cleanup
        env.close()

        print("\n🎉 Real data test completed successfully!")
        return True

    except Exception as e:
        print(f"\n❌ Real data test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_compatibility():
    """Test that data format is compatible between processor and environment."""
    print("\n🔧 Testing Data Format Compatibility")
    print("=" * 50)

    try:
        # Initialize processor
        processor = BinanceProcessor()

        # Small test dataset
        end_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        start_date = (datetime.now() - timedelta(hours=12)).strftime("%Y-%m-%d %H:%M:%S")

        print("📡 Downloading small test dataset...")
        data, price_array, tech_array, time_array, config = processor.run(
            ticker_list=TICKER_LIST[:2],  # Test with first 2 tickers
            start_date=start_date,
            end_date=end_date,
            time_interval='15m',  # Higher timeframe for faster test
            technical_indicator_list=TECHNICAL_INDICATORS_LIST,
            if_vix=False
        )

        print(f"✓ Data shape: Price {price_array.shape}, Tech {tech_array.shape}")

        # Test environment initialization
        env = CryptoEnvCCXT(
            config=config,
            env_params=ENV_PARAMS_CCXT,
            initial_capital=50000
        )

        print("✓ Environment accepts data format correctly")

        # Test reset and step
        state = env.reset()
        action = np.random.uniform(-0.01, 0.01, env.action_dim)
        next_state, reward, done, info = env.step(action)

        print(f"✓ Single step works: reward {reward:.6f}")

        env.close()
        print("✓ Data compatibility test passed!")
        return True

    except Exception as e:
        print(f"❌ Data compatibility test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🚀 CCXT Environment Integration Tests")
    print("=" * 50)

    # Test 1: Data compatibility
    compatibility_success = test_data_compatibility()

    # Test 2: Real data test
    real_data_success = test_with_real_binance_data()

    print("\n" + "=" * 50)
    print("📋 Test Results:")
    print(f"  - Data Compatibility: {'✓ PASSED' if compatibility_success else '❌ FAILED'}")
    print(f"  - Real Data Test: {'✓ PASSED' if real_data_success else '❌ FAILED'}")

    if compatibility_success and real_data_success:
        print("\n🎉 All integration tests passed!")
        print("💡 The CCXT environment is ready for production use!")
        return True
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)