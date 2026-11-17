"""Figures are published with their audited report and input manifest."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from report_study import main

if __name__ == '__main__':
    main()
