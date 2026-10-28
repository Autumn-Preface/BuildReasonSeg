# Isolated launcher contract probes: all side-effect commands are mock functions in this process.
$ErrorActionPreference='Stop'
$reviewRepo=Split-Path -Parent (Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)))
$reviewTool=Join-Path $reviewRepo 'research/cubit_text_pilot'
$reviewHash=(Get-FileHash -LiteralPath (Join-Path $reviewTool 'samples.json') -Algorithm SHA256).Hash.ToLower()
$reviewKeys=@((Get-Content -LiteralPath (Join-Path $reviewTool 'samples.json') -Raw | ConvertFrom-Json).images | ForEach-Object {[pscustomobject]@{key=$_.key}})
function Invoke-RestMethod { param($Uri,$TimeoutSec)
    $global:reviewRequest++
    if($global:reviewMode -like 'new-*' -and $global:reviewRequest -eq 1){throw 'Injected absence before launch'}
    $s=[pscustomobject]@{tool='cubit-text-pilot-01';manifest_sha256=$reviewHash;drafts_root=(Join-Path $reviewTool 'drafts');images=$reviewKeys}
    if($global:reviewMode -match 'qa'){$s.drafts_root=Join-Path $reviewTool 'qa_drafts/review-wrong'}
    if($global:reviewMode -match 'hash'){$s.manifest_sha256='wrong'}
    if($global:reviewMode -match 'tool'){$s.tool='wrong'}
    return $s
}
function Start-Process { param($FilePath,$ArgumentList,$WorkingDirectory,$WindowStyle,$RedirectStandardOutput,$RedirectStandardError,[switch]$PassThru)
    $global:reviewCalls.Add([pscustomobject]@{kind=if($PassThru){'server'}else{'browser'};target=$FilePath;cwd=$WorkingDirectory;window=$WindowStyle;args=$ArgumentList})
    if($PassThru){return [pscustomobject]@{Id=999999;HasExited=$false}}
}
function New-Item { param($ItemType,$Path,[switch]$Force) $global:reviewCalls.Add([pscustomobject]@{kind='directory';target=$Path}) }
function Set-Content { param($LiteralPath,[Parameter(ValueFromPipeline=$true)]$Value) process {$global:reviewCalls.Add([pscustomobject]@{kind='pid';target=$LiteralPath})} }
function Start-Sleep { param($Milliseconds) }
$reviewResults=@()
foreach($reviewCase in @('existing-ok','existing-qa','existing-hash','existing-tool','new-ok','new-qa','new-hash')){
    $global:reviewMode=$reviewCase;$global:reviewRequest=0;$global:reviewCalls=[System.Collections.Generic.List[object]]::new();$reviewFailed=$false;$reviewError=$null
    try {. (Join-Path $reviewTool 'start_review.ps1') | Out-Null} catch {$reviewFailed=$true;$reviewError=$_.Exception.Message}
    $browser=@($global:reviewCalls | Where-Object {$_.kind -eq 'browser'});$pidWrites=@($global:reviewCalls | Where-Object {$_.kind -eq 'pid'})
    $shouldFail=$reviewCase -notmatch '-ok$'
    $passed=if($shouldFail){$reviewFailed -and $browser.Count -eq 0 -and $pidWrites.Count -eq 0}else{-not $reviewFailed -and $browser.Count -eq 1}
    $reviewResults+= [pscustomobject]@{case=$reviewCase;pass=$passed;threw=$reviewFailed;error=$reviewError;calls=@($global:reviewCalls)}
}
$reviewOutput=[pscustomobject]@{all_pass=(@($reviewResults | Where-Object {-not $_.pass}).Count -eq 0);checks=$reviewResults;scope='PowerShell function mocks only; no browser, service, socket, directory, PID or log writes performed by launcher'}
# Qualified cmdlet bypasses only the mock for saving our own independent result.
$reviewOutput | ConvertTo-Json -Depth 8 | Microsoft.PowerShell.Management\Set-Content -LiteralPath (Join-Path $PSScriptRoot 'launcher_results.json') -Encoding utf8
$reviewOutput | ConvertTo-Json -Depth 8
if(-not $reviewOutput.all_pass){exit 1}
