#!/usr/bin/env python3
"""
Test CCXT environment logic with synthetic data of various sizes.
"""

import numpy as np
from environment_CCXT import CryptoEnvCCXT


def test_environment_with_various_sizes():
    """Test environment with different dataset sizes."""
    print("🧪 Testing CCXT Environment with Various Dataset Sizes")
    print("=" * 60)

    test_cases = [
        {"n_steps": 100, "n_cryptos": 2, "lookback": 10, "name": "Small Dataset"},
        {"n_steps": 50, "n_cryptos": 2, "lookback": 5, "name": "Very Small Dataset"},
        {"n_steps": 20, "n_cryptos": 2, "lookback": 3, "name": "Minimal Dataset"},
    ]

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📊 Test Case {i}: {test_case['name']}")
        print(f"  - Data points: {test_case['n_steps']}")
        print(f"  - Cryptos: {test_case['n_cryptos']}")
        print(f"  - Lookback: {test_case['lookback']}")

        try:
            # Create synthetic data
            price_array = np.random.uniform(1000, 50000, (test_case['n_steps'], test_case['n_cryptos']))
            tech_array = np.random.randn(test_case['n_steps'], 5 * test_case['n_cryptos'])  # 5 tech indicators per crypto

            config = {
                'price_array': price_array,
                'tech_array': tech_array,
                'ticker_list': [f'CRYPTO{i}/USDT' for i in range(test_case['n_cryptos'])]
            }

            # Environment parameters
            env_params = {
                'lookback': test_case['lookback'],
                'norm_cash': 1e-6,
                'norm_stocks': 100,
                'norm_tech': 1,
                'norm_reward': 1,
                'norm_action': 1,
            }

            # Initialize environment
            env = CryptoEnvCCXT(
                config=config,
                env_params=env_params,
                initial_capital=10000,
                exchange_name='binance'
            )

            print(f"  ✓ Environment initialized")
            print(f"    - Action dim: {env.action_dim}")
            print(f"    - State dim: {env.state_dim}")
            print(f"    - Max steps: {env.max_step}")

            # Test reset
            state = env.reset()
            print(f"  ✓ Reset successful, state shape: {state.shape}")

            # Test a few steps
            steps_to_test = min(5, env.max_step)
            for step in range(steps_to_test):
                action = np.random.uniform(-0.001, 0.001, env.action_dim)
                state, reward, done, info = env.step(action)

                if step == 0:
                    print(f"  ✓ First step: reward={reward:.6f}")

                if done:
                    print(f"  ✓ Environment done after {step + 1} steps")
                    break

            print(f"  ✓ {test_case['name']} test PASSED")

            env.close()

        except Exception as e:
            print(f"  ❌ {test_case['name']} test FAILED: {e}")
            return False

    return True


def test_edge_cases():
    """Test edge cases and boundary conditions."""
    print("\n🔬 Testing Edge Cases")
    print("=" * 30)

    # Test with single crypto
    print("\n📈 Testing Single Crypto:")
    try:
        price_array = np.random.uniform(30000, 40000, (50, 1))
        tech_array = np.random.randn(50, 5)

        config = {
            'price_array': price_array,
            'tech_array': tech_array,
            'ticker_list': ['BTC/USDT']
        }

        env_params = {'lookback': 5, 'norm_cash': 1e-6, 'norm_stocks': 100, 'norm_tech': 1, 'norm_reward': 1, 'norm_action': 1}

        env = CryptoEnvCCXT(config=config, env_params=env_params, initial_capital=5000)
        state = env.reset()
        action = np.array([0.001])  # Single action
        state, reward, done, info = env.step(action)

        print("  ✓ Single crypto test PASSED")
        env.close()

    except Exception as e:
        print(f"  ❌ Single crypto test FAILED: {e}")
        return False

    # Test with minimal data
    print("\n📉 Testing Minimal Data:")
    try:
        price_array = np.random.uniform(1000, 2000, (10, 2))  # Very small dataset
        tech_array = np.random.randn(10, 10)

        config = {
            'price_array': price_array,
            'tech_array': tech_array,
            'ticker_list': ['ETH/USDT', 'BNB/USDT']
        }

        env_params = {'lookback': 2, 'norm_cash': 1e-6, 'norm_stocks': 100, 'norm_tech': 1, 'norm_reward': 1, 'norm_action': 1}

        env = CryptoEnvCCXT(config=config, env_params=env_params, initial_capital=1000)
        state = env.reset()
        action = np.random.uniform(-0.001, 0.001, 2)
        state, reward, done, info = env.step(action)

        print("  ✓ Minimal data test PASSED")
        env.close()

    except Exception as e:
        print(f"  ❌ Minimal data test FAILED: {e}")
        return False

    return True


def test_trading_logic():
    """Test basic trading logic."""
    print("\n💰 Testing Trading Logic")
    print("=" * 30)

    try:
        # Create deterministic price data for predictable testing
        np.random.seed(42)
        n_steps = 100
        price_array = np.zeros((n_steps, 2))

        # BTC: trending up
        price_array[:, 0] = 40000 + np.cumsum(np.random.normal(10, 100, n_steps))

        # ETH: trending down
        price_array[:, 1] = 3000 - np.cumsum(np.random.normal(5, 50, n_steps))

        tech_array = np.random.randn(n_steps, 10)

        config = {
            'price_array': price_array,
            'tech_array': tech_array,
            'ticker_list': ['BTC/USDT', 'ETH/USDT']
        }

        env_params = {'lookback': 10, 'norm_cash': 1e-6, 'norm_stocks': 100, 'norm_tech': 1, 'norm_reward': 1, 'norm_action': 1}

        env = CryptoEnvCCXT(config=config, env_params=env_params, initial_capital=100000)

        # Test buy action
        state = env.reset()
        initial_portfolio = env.get_portfolio_value()

        # Strong buy signal for first crypto (BTC)
        buy_action = np.array([0.01, 0])  # Buy BTC
        state, reward, done, info = env.step(buy_action)
        portfolio_after_buy = env.get_portfolio_value()

        print(f"  ✓ Initial portfolio: ${initial_portfolio:,.2f}")
        print(f"  ✓ After buy action: ${portfolio_after_buy:,.2f}")

        # Test sell action
        sell_action = np.array([-0.005, 0])  # Sell some BTC
        state, reward, done, info = env.step(sell_action)
        portfolio_after_sell = env.get_portfolio_value()

        print(f"  ✓ After sell action: ${portfolio_after_sell:,.2f}")

        positions = env.get_positions()
        print(f"  ✓ Final positions: {positions}")

        print("  ✓ Trading logic test PASSED")
        env.close()
        return True

    except Exception as e:
        print(f"  ❌ Trading logic test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("🚀 Comprehensive CCXT Environment Testing")
    print("=" * 50)

    # Test 1: Various dataset sizes
    size_test_passed = test_environment_with_various_sizes()

    # Test 2: Edge cases
    edge_test_passed = test_edge_cases()

    # Test 3: Trading logic
    trading_test_passed = test_trading_logic()

    print("\n" + "=" * 50)
    print("📋 Test Summary:")
    print(f"  - Dataset Sizes: {'✓ PASSED' if size_test_passed else '❌ FAILED'}")
    print(f"  - Edge Cases: {'✓ PASSED' if edge_test_passed else '❌ FAILED'}")
    print(f"  - Trading Logic: {'✓ PASSED' if trading_test_passed else '❌ FAILED'}")

    all_passed = size_test_passed and edge_test_passed and trading_test_passed

    if all_passed:
        print("\n🎉 ALL TESTS PASSED!")
        print("💡 CCXT environment implementation is robust and ready!")
    else:
        print("\n❌ SOME TESTS FAILED!")
        print("🔧 Please check the implementation.")

    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)