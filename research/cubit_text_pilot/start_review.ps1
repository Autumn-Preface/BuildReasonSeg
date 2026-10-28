$ErrorActionPreference='Stop'
$pilotRoot=$PSScriptRoot
$pilotRepo=Split-Path -Parent (Split-Path -Parent $pilotRoot)
$pilotPython=Join-Path $pilotRepo '.conda/buildreasonseg-damage-p1a/python.exe'
if(-not (Test-Path -LiteralPath $pilotPython)){throw 'P1A Python not found; no dependencies will be installed.'}
$pilotUrl='http://127.0.0.1:8765'
$pilotSession=$null
try {$pilotSession=Invoke-RestMethod -Uri ($pilotUrl+'/api/session') -TimeoutSec 2} catch {}
if($pilotSession){
    $pilotKeys=@($pilotSession.images | ForEach-Object {$_.key})
    if($pilotKeys.Count -ne 10 -or $pilotKeys -notcontains 'CR1_177' -or $pilotKeys -notcontains 'SP0_794'){throw 'Port 8765 is occupied by a different session. Close that server first.'}
} else {
    $pilotDrafts=Join-Path $pilotRoot 'drafts'
    New-Item -ItemType Directory -Path $pilotDrafts -Force | Out-Null
    $pilotProcess=Start-Process -FilePath $pilotPython -ArgumentList @('-B',('"'+(Join-Path $pilotRoot 'server.py')+'"')) -WorkingDirectory $pilotRepo -WindowStyle Hidden -RedirectStandardOutput (Join-Path $pilotDrafts 'server.stdout.log') -RedirectStandardError (Join-Path $pilotDrafts 'server.stderr.log') -PassThru
    for($pilotAttempt=0;$pilotAttempt -lt 30;$pilotAttempt++){
        try {$pilotSession=Invoke-RestMethod -Uri ($pilotUrl+'/api/session') -TimeoutSec 1;break} catch {Start-Sleep -Milliseconds 500}
        if($pilotProcess.HasExited){throw ('Startup failed. See '+(Join-Path $pilotDrafts 'server.stderr.log'))}
    }
    if(-not $pilotSession){throw 'Local workbench startup timed out; see drafts/server.stderr.log.'}
    $pilotProcess.Id | Set-Content -LiteralPath (Join-Path $pilotDrafts 'server.pid')
}
Start-Process $pilotUrl
Write-Output ('Review workbench: '+$pilotUrl)
