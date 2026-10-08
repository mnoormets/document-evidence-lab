Set-Location -LiteralPath $PSScriptRoot
$taskPython = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) { $taskPython = Join-Path $PSScriptRoot '../shopify-workflow-assistant/.venv/Scripts/python.exe' }
& $taskPython -m uvicorn lab.api:app --host 127.0.0.1 --port 8771
