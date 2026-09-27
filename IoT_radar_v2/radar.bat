@echo off
rem Lanceur local : utilise le venv et la libiio du dossier ProjetS7 (rien d'installé sur le PC).
setlocal
set "HERE=%~dp0"
set "ROOT=%HERE%.."
set "PATH=%ROOT%\tools\libiio\Windows-VS-2022-x64;%PATH%"
set "PYTHONPATH=%HERE%;%PYTHONPATH%"
set "PYTHONIOENCODING=utf-8"
"%ROOT%\.venv\Scripts\python.exe" -m radar %*
