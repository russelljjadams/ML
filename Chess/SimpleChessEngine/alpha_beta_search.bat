@echo off
cd /d "%~dp0"
python alpha_beta_search.py %* 2> error_log.txt