#!/usr/bin/env python3
"""
Скрипт автоматической установки зависимостей для FinRL Crypto
Адаптирован для Python 3.10
"""

import subprocess
import sys
import os

def check_python_version():
    """Проверка версии Python"""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")

    if version.major != 3:
        print("❌ Требуется Python 3.x")
        sys.exit(1)

    if version.minor == 10:
        print("✅ Python 3.10 - рекомендованная версия")
        return "python310"
    elif version.minor >= 11:
        print("✅ Python 3.11+ - поддерживается")
        return "python311+"
    else:
        print("⚠️  Рекомендуется Python 3.10 или выше")
        return "legacy"

def install_requirements(req_file):
    """Установка зависимостей из файла"""
    if not os.path.exists(req_file):
        print(f"❌ Файл {req_file} не найден")
        return False

    print(f"📦 Установка зависимостей из {req_file}...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req_file])
        print("✅ Зависимости успешно установлены")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Ошибка при установке: {e}")
        return False

def check_critical_packages():
    """Проверка критически важных пакетов"""
    critical_packages = [
        "torch",
        "pandas",
        "numpy",
        "binance",
        "matplotlib"
    ]

    print("\n🔍 Проверка критических пакетов:")
    failed = []

    for package in critical_packages:
        try:
            if package == "binance":
                __import__("binance.client")
            else:
                __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            failed.append(package)

    if failed:
        print(f"\n⚠️  Не удалось установить: {', '.join(failed)}")
        print("Попробуйте установить вручную:")
        for pkg in failed:
            if pkg == "binance":
                print(f"pip install python-binance==1.0.19")
            else:
                print(f"pip install {pkg}")
    else:
        print("\n🎉 Все критические пакеты установлены!")

def main():
    """Главная функция"""
    print("🚀 FinRL Crypto - Установка зависимостей\n")

    # Проверка версии Python
    python_type = check_python_version()

    # Выбор файла зависимостей
    if python_type == "python310":
        req_file = "requirements-python310.txt"
    else:
        req_file = "requirements.txt"

    print(f"\n📋 Будет использован файл: {req_file}")

    # Установка зависимостей
    if install_requirements(req_file):
        # Проверка установки
        check_critical_packages()

        print("\n✅ Установка завершена!")
        print("\n📝 Следующие шаги:")
        print("1. Настройте API ключи в config_api.py")
        print("2. Запустите: python 0_dl_trainval_data.py")
        print("3. Запустите: python 1_optimize_cpcv.py")

    else:
        print("\n❌ Установка не удалась")
        sys.exit(1)

if __name__ == "__main__":
    main()