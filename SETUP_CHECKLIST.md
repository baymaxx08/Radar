# Installation Verification Checklist

Complete this checklist to ensure your Corn Cob Classification System is properly installed.

## Pre-Flight Checks

- [ ] Python 3.8+ installed (`python --version` shows 3.8 or higher)
- [ ] Working directory is `c:\Users\Siddhartha Reddy\Desktop\radar`
- [ ] Internet connection (for downloading dependencies)
- [ ] Admin access to install packages (if needed)

## Repository Files

Verify these files exist in your radar directory:

### Core Application Files
- [ ] `app.py` - Flask web server
- [ ] `model.py` - Neural network model
- [ ] `data_loader.py` - CSV data processor
- [ ] `requirements.txt` - Package dependencies
- [ ] `config.py` - Configuration settings

### Startup Scripts
- [ ] `run.py` - Python startup script
- [ ] `run.bat` - Windows batch startup script
- [ ] `test.py` - Validation test script

### Documentation
- [ ] `README.md` - Full documentation
- [ ] `QUICKSTART.md` - Quick start guide
- [ ] `SETUP_CHECKLIST.md` - This file

### Web Interface
- [ ] `templates/index.html` - Web page (check templates folder exists)

### Data Directory
- [ ] `20250319/` folder exists
- [ ] At least 3-6 CSV files in 20250319/ folder:
  - [ ] `202503019_GH_exposed_corn_ear_cls001.csv` (or similar)
  - [ ] `202503019_GH_hidden_corn_ear_cls005.csv` (or similar)
  - [ ] `202503019_GH_stock_cls006.csv` (or similar)

## Dependency Installation

### Step 1: Install Python Packages
```cmd
pip install -r requirements.txt
```

Wait for completion, then verify:

- [ ] `pip show flask` shows Flask installed
- [ ] `pip show tensorflow` shows TensorFlow installed
- [ ] `pip show pandas` shows Pandas installed
- [ ] `pip show numpy` shows NumPy installed
- [ ] `pip show scikit-learn` shows scikit-learn installed

### Step 2: Verify Individual Packages

Run each command to verify:

```cmd
python -c "import flask; print('Flask OK')"
python -c "import tensorflow; print('TensorFlow OK')"
python -c "import pandas; print('Pandas OK')"
python -c "import numpy; print('NumPy OK')"
python -c "import sklearn; print('Scikit-learn OK')"
```

All should print "OK" messages. If any fail, run:
```cmd
pip install [package-name]
```

## Data Verification

Verify your radar data is correctly formatted:

### Check Data Directory
```cmd
dir 20250319\
```
Should list CSV files with "MrmFullScanInfo" data

### Check CSV File Format
Open any CSV file and verify:
- [ ] File contains "Config" lines
- [ ] File contains "MrmFullScanInfo[v1,v2,...,v480]" entries
- [ ] File contains "MrmDetectionListInfo" entries
- [ ] Files are not corrupted (can open in text editor)

## Validation Tests

### Run Validation Suite
```cmd
python test.py
```

Should show:
- [ ] ✓ PASS - Data Loading
- [ ] ✓ PASS - Model Building
- [ ] ✓ PASS - Prediction
- [ ] ✓ PASS - Flask Application

If any tests fail, review the error messages and troubleshoot.

## Network Configuration

### Check Port Availability
Port 5000 must be available:

**Windows:**
```cmd
netstat -ano | findstr :5000
```
Should return nothing (port is free)

If port 5000 is in use, you can change it in `app.py`:
```python
app.run(debug=True, host='0.0.0.0', port=8000)  # Use 8000 instead
```

## System Requirements

Verify your system has adequate resources:

- [ ] CPU: Modern processor (Intel i5/Ryzen 5 or better)
- [ ] RAM: Minimum 4GB, recommended 8GB+
- [ ] Disk Space: 500MB free for dependencies and model
- [ ] GPU: Optional (Nvidia GPU with CUDA for faster training)

## Startup Test

### Launch Application
```cmd
python app.py
```
or double-click `run.bat` on Windows

Watch for:
- [ ] "Loading training data..." message
- [ ] "Loaded X scans" message (should load 50+ scans)
- [ ] "Training model..." message
- [ ] "Epoch X/30" progress display
- [ ] "Model training completed!" message
- [ ] "Running on http://0.0.0.0:5000" message

### Access Web Interface
1. Open web browser
2. Go to: `http://localhost:5000`
3. Verify you see:
   - [ ] "Corn Cob Classification System" title
   - [ ] "Analyze CSV File" section
   - [ ] "Single Scan Analysis" section
   - [ ] "Model Status: ✓ Model Ready" (or "Loading...")

## Functionality Tests

### Test 1: CSV File Upload
1. [ ] Use "Choose File" button
2. [ ] Select a CSV from 20250319/ folder
3. [ ] Click "Analyze File"
4. [ ] Wait for results (should show predictions)
5. [ ] Verify you see class distribution

### Test 2: Single Scan Classification
1. [ ] Copy 480 comma-separated values from any CSV file
2. [ ] Paste into the textarea
3. [ ] Click "Classify Scan"
4. [ ] See prediction with confidence score
5. [ ] View probability breakdown

## Performance Benchmarks

After successful startup, check these metrics:

- [ ] Model training completed in < 10 minutes
- [ ] Single scan prediction: < 100ms
- [ ] CSV file analysis: < 1 second for 50 scans
- [ ] Web interface loads in < 2 seconds
- [ ] All CSS/styling displays correctly

## Troubleshooting Checklist

If something doesn't work, check:

- [ ] Python interpreter is Python 3.8+ (run `python --version`)
- [ ] All files listed above exist in the project directory
- [ ] No special characters in file/folder names
- [ ] CSV files are readable (not corrupted)
- [ ] Port 5000 is not blocked by firewall
- [ ] TensorFlow is fully installed (`pip install --upgrade tensorflow`)
- [ ] Sufficient disk space available (>500MB)
- [ ] System isn't in power-saving mode (can interrupt model training)

## Final Sign-Off

When all checkboxes above are complete:

- [ ] All files present and correct ✓
- [ ] All dependencies installed ✓
- [ ] Data files verified ✓
- [ ] Validation tests passed ✓
- [ ] Application starts successfully ✓
- [ ] Web interface accessible ✓
- [ ] Predictions working ✓

## You're Ready!

🎉 Your Corn Cob Classification System is ready to use!

### Next Steps:
1. Open `http://localhost:5000` in your browser
2. Upload a CSV file from the 20250319 folder
3. View predictions and analysis results
4. Start classifying radar data!

## Need Help?

If you encounter issues:
1. Review the error message carefully
2. Check the README.md for detailed explanations
3. Run `python test.py` to validate setup
4. Check console output for error details
5. Verify all dependencies are installed

Good luck with your corn cob classification project! 🌽🛰️
