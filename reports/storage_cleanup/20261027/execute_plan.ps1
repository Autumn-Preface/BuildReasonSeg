param([Parameter(Mandatory=$true)][string]$ExpectedPlanHash)
$ErrorActionPreference='Stop'
$maintenanceRoot=[IO.Path]::GetFullPath((Get-Item -LiteralPath $PSScriptRoot).Parent.Parent.Parent.FullName)
if($maintenanceRoot -ne 'C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg'){throw 'Unexpected absolute workspace root'}
$maintenancePlanPath=Join-Path $PSScriptRoot 'plan.json'
if((Get-FileHash -LiteralPath $maintenancePlanPath -Algorithm SHA256).Hash.ToLower() -ne $ExpectedPlanHash.ToLower()){throw 'Reviewed plan hash differs'}
$maintenancePlan=Get-Content -LiteralPath $maintenancePlanPath -Raw -Encoding UTF8 | ConvertFrom-Json
if([IO.Path]::GetFullPath($maintenancePlan.scope_root) -ne $maintenanceRoot){throw 'Plan scope differs'}
if(Test-Path -LiteralPath (Join-Path $PSScriptRoot 'deletion_log.jsonl')){throw 'Prior execution log exists; do not rerun blindly'}
$maintenanceAllowedTrees=@(
 '.tmp\demo_r1_57_final','.tmp\demo_r1_final_2334','.tmp\demo_r1_fake_evidence',
 '.tmp\damage_p1a_tests_04','.tmp\damage_p1a_tests_05','.tmp\damage_p1a_tests_06',
 'artifacts\damage_p1a\pip_cache'
) | ForEach-Object {[IO.Path]::GetFullPath((Join-Path $maintenanceRoot $_))}
$maintenanceAllowedFiles=@(
 'artifacts\damage_p1a\source\1HqS0rMWgwsy70TbaZYFqwzgFxLL_hp_b.html',
 'artifacts\damage_p1a\source\1MdeKlUkC_8rsav8kXPn8Y6jfhLXOKBz3.html',
 'artifacts\damage_p1a\source\1sffbt7yZ5jnQtkK4c2hg1l0XtOBtg3OD.html',
 'artifacts\damage_p1a\source\1WZKYslWDqXUl-n4mp23j48JdR9wZ3J6T.html',
 'artifacts\damage_p1a\duplicate_samples\val_SP0_818.JPG',
 'artifacts\damage_p1a\duplicate_samples\val_SP0_1102.JPG'
) | ForEach-Object {[IO.Path]::GetFullPath((Join-Path $maintenanceRoot $_))}
function Assert-MaintenancePath([string]$candidate,[bool]$Directory=$false){
 $resolved=[IO.Path]::GetFullPath($candidate)
 if(-not $resolved.StartsWith($maintenanceRoot+'\',[StringComparison]::OrdinalIgnoreCase)){throw "Outside workspace: $resolved"}
 $allowed=(!$Directory -and ($maintenanceAllowedFiles -contains $resolved))
 foreach($tree in $maintenanceAllowedTrees){if($resolved -eq $tree -or $resolved.StartsWith($tree+'\',[StringComparison]::OrdinalIgnoreCase)){$allowed=$true}}
 if(-not $allowed){throw "Not in hard-coded reviewed allowlist: $resolved"}
 $ancestor=$resolved
 while($ancestor -ne $maintenanceRoot){
  $item=Get-Item -LiteralPath $ancestor -Force -ErrorAction Stop
  if(($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0){throw "Reparse point refused: $ancestor"}
  $ancestor=[IO.Path]::GetDirectoryName($ancestor)
 }
 return $resolved
}
# Exclude this PowerShell maintenance process. No project Python/download worker may be active.
$maintenanceActive=@(Get-CimInstance Win32_Process | Where-Object {
 $_.Name -match '^(pythonw?|conda|aria2c|wget|curl)\.exe$' -and
 (($_.ExecutablePath -and $_.ExecutablePath.StartsWith($maintenanceRoot+'\',[StringComparison]::OrdinalIgnoreCase)) -or
  ($_.CommandLine -and ($_.CommandLine.IndexOf($maintenanceRoot,[StringComparison]::OrdinalIgnoreCase) -ge 0 -or
                       $_.CommandLine.IndexOf('C:\D\DeepSeekHarness\delivery\BuildReasonSeg_',[StringComparison]::OrdinalIgnoreCase) -ge 0)))
} | Select-Object ProcessId,Name,CommandLine)
ConvertTo-Json -InputObject @($maintenanceActive) -Depth 4 | Set-Content -LiteralPath (Join-Path $PSScriptRoot 'process_check_before_delete.json') -Encoding UTF8
if($maintenanceActive.Count -gt 0){throw 'Active project worker found; no deletion'}
# Verify the complete file allowlist and retained duplicate originals BEFORE removing even one file.
foreach($row in $maintenancePlan.deletions){
 $path=Assert-MaintenancePath $row.path
 $item=Get-Item -LiteralPath $path -Force
 if($item.PSIsContainer -or $item.Length -ne $row.bytes){throw "Type/size drift: $path"}
 if((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLower() -ne $row.sha256){throw "Content drift: $path"}
 if($row.retained_copy){
  $keep=[IO.Path]::GetFullPath($row.retained_copy.path)
  if(-not $keep.StartsWith($maintenanceRoot+'\',[StringComparison]::OrdinalIgnoreCase)){throw 'Invalid retained copy path'}
  if((Get-FileHash -LiteralPath $keep -Algorithm SHA256).Hash.ToLower() -ne $row.sha256){throw 'Retained original drift'}
 }
}
$maintenanceDisk=Get-PSDrive -Name C
@{free_bytes=$maintenanceDisk.Free;total_bytes=($maintenanceDisk.Free+$maintenanceDisk.Used);at=(Get-Date).ToString('o');plan_sha256=$ExpectedPlanHash} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $PSScriptRoot 'disk_immediately_before.json') -Encoding UTF8
foreach($row in $maintenancePlan.deletions){
 # Literal single-file removal only. No wildcard, shell chaining, or recursive deletion.
 $path=Assert-MaintenancePath $row.path
 if((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLower() -ne $row.sha256){throw "Late content drift: $path"}
 Remove-Item -LiteralPath $path -Force -ErrorAction Stop
 @{path=$path;bytes=$row.bytes;sha256=$row.sha256;category=$row.category;status='DELETED';at=(Get-Date).ToString('o')} | ConvertTo-Json -Compress | Add-Content -LiteralPath (Join-Path $PSScriptRoot 'deletion_log.jsonl') -Encoding UTF8
}
$maintenanceEmptyRemoved=0
foreach($dir in $maintenancePlan.empty_directories_after_files){
 if(-not (Test-Path -LiteralPath $dir)){continue}
 $resolved=Assert-MaintenancePath $dir $true
 # Non-recursive .NET deletion atomically fails if any unknown/new child remains.
 if(@(Get-ChildItem -LiteralPath $resolved -Force).Count -eq 0){[IO.Directory]::Delete($resolved,$false);$maintenanceEmptyRemoved++}
}
$maintenanceDisk=Get-PSDrive -Name C
@{free_bytes=$maintenanceDisk.Free;total_bytes=($maintenanceDisk.Free+$maintenanceDisk.Used);at=(Get-Date).ToString('o');empty_directories_removed=$maintenanceEmptyRemoved} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $PSScriptRoot 'disk_immediately_after.json') -Encoding UTF8
Write-Output "Deleted exactly $($maintenancePlan.total_files) reviewed files, $($maintenancePlan.total_logical_bytes) logical bytes."
