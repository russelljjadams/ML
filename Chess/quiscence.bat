@echo off
cd /d "%~dp0"
python quiscence.py %* 2> error_log.txt