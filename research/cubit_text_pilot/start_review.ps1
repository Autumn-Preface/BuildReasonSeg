$ErrorActionPreference='Stop'
$pilotRoot=$PSScriptRoot
$pilotRepo=Split-Path -Parent (Split-Path -Parent $pilotRoot)
$pilotPython=Join-Path $pilotRepo '.conda/buildreasonseg-damage-p1a/python.exe'
if(-not (Test-Path -LiteralPath $pilotPython)){throw 'P1A Python not found; no dependencies will be installed.'}
$pilotUrl='http://127.0.0.1:8765'
$pilotDrafts=Join-Path $pilotRoot 'drafts'
$pilotManifestHash=(Get-FileHash -LiteralPath (Join-Path $pilotRoot 'samples.json') -Algorithm SHA256).Hash.ToLower()
$pilotSession=$null
$pilotProcess=$null
try {$pilotSession=Invoke-RestMethod -Uri ($pilotUrl+'/api/session') -TimeoutSec 2} catch {}
if($pilotSession){
    $pilotKeys=@($pilotSession.images | ForEach-Object {$_.key})
    if($pilotSession.tool -ne 'cubit-text-pilot-01' -or $pilotSession.manifest_sha256 -ne $pilotManifestHash -or $pilotSession.drafts_root -ne $pilotDrafts -or $pilotKeys.Count -ne 10){throw 'Port 8765 is occupied by a different session. Close that server first.'}
} else {
    New-Item -ItemType Directory -Path $pilotDrafts -Force | Out-Null
    $pilotProcess=Start-Process -FilePath $pilotPython -ArgumentList @('-B',('"'+(Join-Path $pilotRoot 'server.py')+'"')) -WorkingDirectory $pilotRepo -WindowStyle Hidden -RedirectStandardOutput (Join-Path $pilotDrafts 'server.stdout.log') -RedirectStandardError (Join-Path $pilotDrafts 'server.stderr.log') -PassThru
    for($pilotAttempt=0;$pilotAttempt -lt 30;$pilotAttempt++){
        try {$pilotSession=Invoke-RestMethod -Uri ($pilotUrl+'/api/session') -TimeoutSec 1;break} catch {Start-Sleep -Milliseconds 500}
        if($pilotProcess.HasExited){throw ('Startup failed. See '+(Join-Path $pilotDrafts 'server.stderr.log'))}
    }
    if(-not $pilotSession){throw 'Local workbench startup timed out; see drafts/server.stderr.log.'}
}
$pilotKeys=@($pilotSession.images | ForEach-Object {$_.key})
if($pilotSession.tool -ne 'cubit-text-pilot-01' -or $pilotSession.manifest_sha256 -ne $pilotManifestHash -or $pilotSession.drafts_root -ne $pilotDrafts -or $pilotKeys.Count -ne 10){throw 'Session identity mismatch. Browser was not opened and no PID was written.'}
if($pilotProcess){$pilotProcess.Id | Set-Content -LiteralPath (Join-Path $pilotDrafts 'server.pid')}
Start-Process $pilotUrl
Write-Output ('Review workbench: '+$pilotUrl)
