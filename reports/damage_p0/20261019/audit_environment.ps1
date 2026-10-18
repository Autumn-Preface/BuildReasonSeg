$ErrorActionPreference = 'Continue'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '../../..')).Path
$out = Join-Path $repo 'evaluation/damage_p0/environment_facts.json'
function Probe($Name, [scriptblock]$Action) {
    try { $value = & $Action 2>&1 | Out-String; return @{name=$Name; output=$value.Trim(); success=$?} }
    catch { return @{name=$Name; output=$_.Exception.Message; success=$false} }
}
$facts = [ordered]@{
    captured_at = [DateTimeOffset]::Now.ToString('o')
    mode = 'READ_ONLY_NO_INSTALL_NO_MODEL_LOAD'
    probes = @(
        (Probe 'windows' {Get-CimInstance Win32_OperatingSystem | Select-Object Caption,Version,BuildNumber,OSArchitecture,TotalVisibleMemorySize,FreePhysicalMemory | ConvertTo-Json}),
        (Probe 'cpu' {Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors | ConvertTo-Json}),
        (Probe 'memory' {Get-CimInstance Win32_ComputerSystem | Select-Object TotalPhysicalMemory | ConvertTo-Json}),
        (Probe 'disks' {Get-CimInstance Win32_LogicalDisk -Filter 'DriveType=3' | Select-Object DeviceID,Size,FreeSpace | ConvertTo-Json}),
        (Probe 'gpu_driver' {& "$env:WINDIR/System32/nvidia-smi.exe" -q | Select-String 'NVIDIA-SMI|CUDA Version|Driver Version|CUDA UMD Version'}),
        (Probe 'gpu_query' {& "$env:WINDIR/System32/nvidia-smi.exe" --query-gpu=name,memory.total,driver_version --format=csv,noheader}),
        (Probe 'wsl_version' {& "$env:WINDIR/System32/wsl.exe" --version}),
        (Probe 'wsl_status' {& "$env:WINDIR/System32/wsl.exe" --status}),
        (Probe 'wsl_distros' {& "$env:WINDIR/System32/wsl.exe" --list --verbose}),
        (Probe 'conda_envs' {& 'C:/D/anaconda3/Scripts/conda.exe' env list --json}),
        (Probe 'conda_version' {& 'C:/D/anaconda3/Scripts/conda.exe' --version}),
        (Probe 'git_path' {Get-Command git -All | Select-Object Source | ConvertTo-Json}),
        (Probe 'nvcc_path' {Get-Command nvcc -All -ErrorAction Stop | Select-Object Source | ConvertTo-Json}),
        (Probe 'cache_paths' {Get-ChildItem Env: | Where-Object {$_.Name -in @('HF_HOME','HUGGINGFACE_HUB_CACHE','TORCH_HOME','PIP_CACHE_DIR','CONDA_PKGS_DIRS')} | Select-Object Name,Value | ConvertTo-Json})
    )
}
$facts | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $out -Encoding UTF8
Write-Output $out
