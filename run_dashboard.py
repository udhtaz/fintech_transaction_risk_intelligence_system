"""
Streamlit Dashboard Runner for Fintech Transaction Risk Intelligence System
"""

import streamlit.web.cli as stcli
import sys
import os

if __name__ == "__main__":
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dashboard_path = os.path.join(dir_path, "dashboard", "main.py")
    
    sys.argv = ["streamlit", "run", dashboard_path, "--server.port=8501"]
    sys.exit(stcli.main())
