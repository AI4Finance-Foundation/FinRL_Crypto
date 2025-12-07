#!/usr/bin/env python3
"""
Test script for the new CCXT environment implementation.
This script tests the basic functionality of CryptoEnvCCXT.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from config_main import TICKER_LIST, ENV_PARAMS_CCXT, TRADING_PARAMS
from environment_CCXT import CryptoEnvCCXT


def create_test_data():
    """Create synthetic price and technical indicator data for testing."""
    print("Creating test data...")

    # Generate synthetic price data
    n_steps = 1000
    n_cryptos = len(TICKER_LIST)

    # Create synthetic price paths
    price_array = np.zeros((n_steps, n_cryptos))
    base_prices = [40000, 3000]  # BTC, ETH base prices

    for i in range(n_cryptos):
        # Random walk with drift
        returns = np.random.normal(0.0001, 0.02, n_steps)
        price_path = base_prices[i] * np.exp(np.cumsum(returns))
        price_array[:, i] = price_path

    # Create synthetic technical indicators
    tech_features = 10  # Number of technical indicators
    tech_array = np.random.randn(n_steps, tech_features * n_cryptos)

    # Create config dict
    config = {
        'price_array': price_array,
        'tech_array': tech_array,
        'ticker_list': TICKER_LIST
    }

    return config


def test_environment_initialization():
    """Test that the CCXT environment can be initialized properly."""
    print("\n=== Testing Environment Initialization ===")

    try:
        # Create test data
        config = create_test_data()

        # Initialize environment
        env = CryptoEnvCCXT(
            config=config,
            env_params=ENV_PARAMS_CCXT,
            initial_capital=TRADING_PARAMS['initial_capital'],
            buy_cost_pct=TRADING_PARAMS['buy_cost_pct'],
            sell_cost_pct=TRADING_PARAMS['sell_cost_pct'],
            gamma=TRADING_PARAMS['gamma'],
            exchange_name='binance'
        )

        print(f"✓ Environment initialized successfully")
        print(f"  - Action dimension: {env.action_dim}")
        print(f"  - State dimension: {env.state_dim}")
        print(f"  - Max steps: {env.max_step}")
        print(f"  - Number of cryptos: {env.crypto_num}")
        print(f"  - Initial cash: ${env.initial_cash:,.2f}")

        return env

    except Exception as e:
        print(f"✗ Environment initialization failed: {e}")
        return None


def test_environment_reset(env):
    """Test the environment reset functionality."""
    print("\n=== Testing Environment Reset ===")

    try:
        state = env.reset()

        print(f"✓ Environment reset successfully")
        print(f"  - State shape: {state.shape}")
        print(f"  - Initial portfolio value: ${env.get_portfolio_value():,.2f}")

        return state

    except Exception as e:
        print(f"✗ Environment reset failed: {e}")
        return None


def test_environment_step(env):
    """Test the environment step functionality."""
    print("\n=== Testing Environment Step ===")

    try:
        # Take a random action
        action = np.random.uniform(-0.1, 0.1, env.action_dim)

        # Step through environment
        next_state, reward, done, info = env.step(action)

        print(f"✓ Environment step completed successfully")
        print(f"  - Action taken: {action}")
        print(f"  - Reward: {reward:.6f}")
        print(f"  - Done: {done}")
        print(f"  - Portfolio value: ${env.get_portfolio_value():,.2f}")
        print(f"  - Current positions: {env.get_positions()}")

        return next_state, reward, done

    except Exception as e:
        print(f"✗ Environment step failed: {e}")
        return None, None, None


def test_multiple_steps(env, n_steps=10):
    """Test multiple steps through the environment."""
    print(f"\n=== Testing Multiple Steps ({n_steps} steps) ===")

    try:
        env.reset()
        total_reward = 0
        portfolio_values = []

        for step in range(n_steps):
            action = np.random.uniform(-0.05, 0.05, env.action_dim)
            state, reward, done, _ = env.step(action)

            total_reward += reward
            portfolio_value = env.get_portfolio_value()
            portfolio_values.append(portfolio_value)

            if step % 5 == 0:
                print(f"  Step {step}: Portfolio ${portfolio_value:,.2f}, Reward {reward:.6f}")

            if done:
                print(f"  Environment finished after {step + 1} steps")
                break

        print(f"✓ Multiple steps completed successfully")
        print(f"  - Total reward: {total_reward:.6f}")
        print(f"  - Initial portfolio: ${portfolio_values[0]:,.2f}")
        print(f"  - Final portfolio: ${portfolio_values[-1]:,.2f}")
        print(f"  - Return: {(portfolio_values[-1] / portfolio_values[0] - 1) * 100:.2f}%")

        return True

    except Exception as e:
        print(f"✗ Multiple steps failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🚀 Testing CCXT Environment Implementation")
    print("=" * 50)

    # Test 1: Environment initialization
    env = test_environment_initialization()
    if env is None:
        print("\n❌ Initialization failed. Stopping tests.")
        return False

    # Test 2: Environment reset
    state = test_environment_reset(env)
    if state is None:
        print("\n❌ Reset failed. Stopping tests.")
        return False

    # Test 3: Single step
    next_state, reward, done = test_environment_step(env)
    if next_state is None:
        print("\n❌ Single step failed. Stopping tests.")
        return False

    # Test 4: Multiple steps
    success = test_multiple_steps(env)

    # Cleanup
    env.close()

    print("\n" + "=" * 50)
    if success:
        print("🎉 All tests passed! CCXT environment is working correctly.")
        return True
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)