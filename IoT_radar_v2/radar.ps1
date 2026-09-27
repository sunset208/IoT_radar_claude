# Lanceur PowerShell équivalent à radar.bat
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$root = Split-Path -Parent $here
$env:PATH = "$root\tools\libiio\Windows-VS-2022-x64;$env:PATH"
$env:PYTHONPATH = "$here;$env:PYTHONPATH"
$env:PYTHONIOENCODING = "utf-8"
& "$root\.venv\Scripts\python.exe" -m radar @args
exit $LASTEXITCODE
