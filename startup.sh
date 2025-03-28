#!/bin/bash
pip install -r requirements.txt
streamlit run app.py --server.address=0.0.0.0 --server.port=8000
