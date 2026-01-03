import os
import shutil
import glob

site_packages = os.path.join(os.getcwd(), ".venv", "Lib", "site-packages")
print(f"Scanning {site_packages}...")

matched = glob.glob(os.path.join(site_packages, "*aleido*"))
for path in matched:
    print(f"Removing: {path}")
    try:
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)
        print("Success.")
    except Exception as e:
        print(f"Failed: {e}")
