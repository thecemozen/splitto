@echo off
chcp 65001 >nul
rem Ses dosyalarini (veya bir klasoru) bu dosyanin uzerine surukleyip birakin.
rem Stem'ler her sarkinin yanina "<sarki adi> - Stems" klasorune yazilir.
if "%~1"=="" (
  echo Ses dosyalarini bu dosyanin uzerine surukleyip birakin.
  pause
  exit /b 1
)
python "%~dp0..\splitto\scripts\kurulum.py" || (pause & exit /b 1)
"%USERPROFILE%\.stems-venv\Scripts\python.exe" "%~dp0..\splitto\scripts\stems.py" %*
pause
