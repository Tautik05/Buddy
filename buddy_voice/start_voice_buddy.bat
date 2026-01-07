@echo off
echo Starting Dynamic Voice BUDDY...
echo.
echo Make sure you have:
echo - Microphone connected and working
echo - Speakers/headphones connected
echo - Internet connection for speech recognition
echo.
pause
cd /d "%~dp0"
python dynamic_voice_buddy.py
pause