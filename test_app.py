"""
Quick test script to verify Flask app setup
"""
import sys
import os

def test_imports():
    """Test if all required packages can be imported"""
    print("Testing imports...")

    try:
        import flask
        print(f"✅ Flask {flask.__version__}")
    except ImportError as e:
        print(f"❌ Flask import failed: {e}")
        return False

    try:
        import plotly
        print(f"✅ Plotly {plotly.__version__}")
    except ImportError as e:
        print(f"❌ Plotly import failed: {e}")
        return False

    try:
        import pandas
        print(f"✅ Pandas {pandas.__version__}")
    except ImportError as e:
        print(f"❌ Pandas import failed: {e}")
        return False

    try:
        from binance.client import Client
        print(f"✅ Binance client")
    except ImportError as e:
        print(f"❌ Binance import failed: {e}")
        return False

    return True

def test_modules():
    """Test if our custom modules can be imported"""
    print("\nTesting custom modules...")

    try:
        from chart_generator import generate_chart_data, get_chart_metadata
        print("✅ chart_generator module")
    except ImportError as e:
        print(f"❌ chart_generator import failed: {e}")
        return False

    try:
        from data_updater import update_data_files, parse_filename
        print("✅ data_updater module")
    except ImportError as e:
        print(f"❌ data_updater import failed: {e}")
        return False

    try:
        import app
        print("✅ app module")
    except ImportError as e:
        print(f"❌ app import failed: {e}")
        return False

    return True

def test_data_directory():
    """Test if data directory exists and has CSV files"""
    print("\nTesting data directory...")

    data_dir = os.path.join(os.path.dirname(__file__), 'data')

    if not os.path.exists(data_dir):
        print(f"❌ Data directory not found: {data_dir}")
        return False

    print(f"✅ Data directory exists: {data_dir}")

    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]

    if not csv_files:
        print("⚠️  Warning: No CSV files found in data directory")
        return True

    print(f"✅ Found {len(csv_files)} CSV files:")
    for csv_file in csv_files:
        print(f"   - {csv_file}")

    return True

def test_chart_generation():
    """Test if chart generation works"""
    print("\nTesting chart generation...")

    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]

    if not csv_files:
        print("⚠️  Skipping chart generation test (no CSV files)")
        return True

    try:
        from chart_generator import generate_chart_data
        import re

        # Use first CSV file
        test_file = csv_files[0]
        filepath = os.path.join(data_dir, test_file)

        # Parse symbol and timeframe from filename
        pattern = r'^([A-Z]+)_(\d+[mhd])_data\.csv$'
        match = re.match(pattern, test_file)

        if match:
            symbol = match.group(1)
            timeframe = match.group(2)

            print(f"   Testing with {test_file} ({symbol} - {timeframe})")

            chart_json = generate_chart_data(filepath, symbol, timeframe, 50)

            if chart_json:
                print(f"✅ Chart generation successful (JSON length: {len(chart_json)} chars)")
                return True
            else:
                print("❌ Chart generation returned empty result")
                return False
        else:
            print(f"⚠️  Could not parse filename: {test_file}")
            return True

    except Exception as e:
        print(f"❌ Chart generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("="*60)
    print("Flask Web Monitor - System Test")
    print("="*60)

    tests = [
        ("Package Imports", test_imports),
        ("Custom Modules", test_modules),
        ("Data Directory", test_data_directory),
        ("Chart Generation", test_chart_generation)
    ]

    results = []

    for test_name, test_func in tests:
        result = test_func()
        results.append((test_name, result))

    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)

    all_passed = True
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
        if not result:
            all_passed = False

    print("="*60)

    if all_passed:
        print("\n🎉 All tests passed! You can now run the Flask app:")
        print("   .venv\\Scripts\\python.exe app.py")
        print("   or")
        print("   run_monitor.bat")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        return 1

if __name__ == '__main__':
    sys.exit(main())
