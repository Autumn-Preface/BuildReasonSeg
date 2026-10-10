param([Parameter(Mandatory=$true)][string]$Root,[Parameter(Mandatory=$true)][string]$EvidenceFile)
$ErrorActionPreference='Stop'
$taskResolved=(Resolve-Path -LiteralPath $Root).Path
$taskAllowed=@('C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Demo_V2_task8f_staging_v1','C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Demo_V2')
if($taskResolved -notin $taskAllowed){throw 'Only the authorized new V2 root is allowed'}
$taskLaunch=Join-Path $taskResolved '_ui\launch.ps1'
$taskTokens=$null;$taskErrors=$null
[System.Management.Automation.Language.Parser]::ParseFile($taskLaunch,[ref]$taskTokens,[ref]$taskErrors) | Out-Null
if($taskErrors.Count -ne 0){throw 'Native launcher parse failed'}
function global:Start-Process {
 param([string]$FilePath,[string[]]$ArgumentList,[string]$WorkingDirectory,[string]$WindowStyle)
 $global:task8fCapturedStart=@{FilePath=$FilePath;ArgumentList=$ArgumentList;WorkingDirectory=$WorkingDirectory;WindowStyle=$WindowStyle}
}
# Actual native launcher environment probe executes; only its final new-process dispatch is intercepted.
$global:task8fCapturedStart=$null
& $taskLaunch
if($null -eq $global:task8fCapturedStart){throw 'Launcher did not reach process dispatch'}
$taskCapture=$global:task8fCapturedStart
if($taskCapture.WorkingDirectory -ne $taskResolved -or $taskCapture.WindowStyle -ne 'Hidden'){throw 'Launcher root/window contract failed'}
if($taskCapture.ArgumentList.Count -ne 2 -or $taskCapture.ArgumentList[0] -ne '-B' -or $taskCapture.ArgumentList[1] -ne ('"'+(Join-Path $taskResolved 'BuildReasonSeg_Demo.py')+'"')){throw 'Launcher quoting contract failed'}
$taskBom=[System.IO.File]::ReadAllBytes($taskLaunch)
if($taskBom[0] -ne 239 -or $taskBom[1] -ne 187 -or $taskBom[2] -ne 191){throw 'Windows PowerShell UTF8 BOM missing'}
$taskResult=@{pass=$true;root=$taskResolved;native_powershell=$PSVersionTable.PSVersion.ToString();parse_errors=0;utf8_bom=$true;actual_environment_probe=$true;dispatch=$taskCapture;dispatch_intercepted=$true;model_inference=0}
$taskJson=$taskResult | ConvertTo-Json -Depth 5
[System.IO.File]::WriteAllText($EvidenceFile,$taskJson+[Environment]::NewLine,[System.Text.UTF8Encoding]::new($false))
Write-Output $taskJson
