#!/usr/bin/env python3
"""
Тестовый скрипт для загрузки данных с Binance
Использует актуальные даты и небольшое количество данных для быстрой проверки
"""

from processor_Binance import BinanceProcessor
import pandas as pd
from datetime import datetime, timedelta

def test_data_download():
    """Тест загрузки данных с актуальными параметрами"""

    print("🚀 Тестовая загрузка данных с Binance")
    print("=" * 50)

    # Актуальные параметры для теста
    ticker_list = ['BTCUSDT', 'ETHUSDT']
    timeframe = '5m'  # 5-минутные свечи

    # Последние 2 дня данных
    end_date = datetime.now()
    start_date = end_date - timedelta(days=2)

    start_date_str = start_date.strftime('%Y-%m-%d %H:%M:%S')
    end_date_str = end_date.strftime('%Y-%m-%d %H:%M:%S')

    technical_indicator_list = [
        'open', 'high', 'low', 'close', 'volume',
        'macd', 'rsi', 'cci', 'dx'
    ]

    print(f"📊 Параметры:")
    print(f"   Тикеры: {ticker_list}")
    print(f"   Таймфрейм: {timeframe}")
    print(f"   Период: {start_date_str} - {end_date_str}")
    print(f"   Индикаторы: {len(technical_indicator_list)} шт.")
    print()

    try:
        # Создаем процессор
        processor = BinanceProcessor()

        # Загружаем данные
        print("📥 Загрузка данных...")
        data, price_array, tech_array, time_array, config = processor.run(
            ticker_list=ticker_list,
            start_date=start_date_str,
            end_date=end_date_str,
            time_interval=timeframe,
            technical_indicator_list=technical_indicator_list,
            if_vix=False
        )

        print("✅ Данные успешно загружены!")
        print()

        # Анализ данных
        print("📊 Анализ загруженных данных:")
        print(f"   Размер DataFrame: {data.shape}")
        print(f"   Количество тикеров: {len(data['tic'].unique())}")
        print(f"   Период данных: {data.index.min()} - {data.index.max()}")
        print(f"   Ценовой массив: {price_array.shape}")
        print(f"   Массив индикаторов: {tech_array.shape}")
        print()

        # Проверка данных по тикерам
        print("📈 Данные по тикерам:")
        for ticker in data['tic'].unique():
            ticker_data = data[data['tic'] == ticker]
            print(f"   {ticker}: {len(ticker_data)} записей")

            if len(ticker_data) > 0:
                latest_price = ticker_data['close'].iloc[-1]
                print(f"      Последняя цена: ${latest_price:,.2f}")

                # Проверка технических индикаторов
                if 'rsi' in ticker_data.columns:
                    latest_rsi = ticker_data['rsi'].iloc[-1]
                    print(f"      Последний RSI: {latest_rsi:.2f}")

                if 'macd' in ticker_data.columns:
                    latest_macd = ticker_data['macd'].iloc[-1]
                    print(f"      Последний MACD: {latest_macd:.4f}")

        print()
        print("🎉 Тест успешно завершен!")
        print("📁 Данные готовы для использования в моделях")

        # Сохраняем тестовые данные
        test_file = "test_data.pkl"
        with open(test_file, 'wb') as f:
            pickle.dump({
                'data': data,
                'price_array': price_array,
                'tech_array': tech_array,
                'time_array': time_array
            }, f)

        print(f"💾 Тестовые данные сохранены в: {test_file}")

        return True

    except Exception as e:
        print(f"❌ Ошибка при загрузке данных: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import pickle
    test_data_download()