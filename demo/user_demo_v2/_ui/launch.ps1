$ErrorActionPreference='Stop'
try {
[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new($false)
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONUTF8='1'
$env:PYTHONIOENCODING='utf-8'
$demoRoot=Split-Path -Parent $PSScriptRoot
$demoProbe=Join-Path $PSScriptRoot 'environment.py'
$demoConfig=Get-Content -LiteralPath (Join-Path $PSScriptRoot 'runtime.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$demoCandidates=[System.Collections.Generic.List[string]]::new()
if($env:BUILDREASONSEG_PYTHON){$demoCandidates.Add($env:BUILDREASONSEG_PYTHON)}
$demoCandidates.Add((Join-Path $demoRoot 'runtime\python.exe'))
if($demoConfig.validated_python){$demoCandidates.Add($demoConfig.validated_python)}
$demoPathPython=Get-Command python -ErrorAction SilentlyContinue
if($demoPathPython){$demoCandidates.Add($demoPathPython.Source)}
$demoSelected=$null
foreach($demoPython in ($demoCandidates | Select-Object -Unique)){
    if(-not (Test-Path -LiteralPath $demoPython -PathType Leaf)){continue}
    # environment.py creates private caches before any runtime import; never install anything.
    $ErrorActionPreference='Continue'
    $demoProbeOutput=& $demoPython -B $demoProbe 2>&1
    $demoProbeExit=$LASTEXITCODE
    $ErrorActionPreference='Stop'
    if($demoProbeExit -eq 0){$demoSelected=$demoPython;break}
}
if(-not $demoSelected){
    Add-Type -AssemblyName PresentationFramework
    [System.Windows.MessageBox]::Show('未找到兼容的 Python 运行环境。请指定已配置好的 BUILDREASONSEG_PYTHON 后重新启动。本程序不会安装依赖。','BuildReasonSeg Demo') | Out-Null
    exit 1
}
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONUTF8='1'
$env:PYTHONIOENCODING='utf-8'
$demoApp=Join-Path $demoRoot 'BuildReasonSeg_Demo.py'
Start-Process -FilePath $demoSelected -ArgumentList @('-B',('"'+$demoApp+'"')) -WorkingDirectory $demoRoot -WindowStyle Hidden
} catch {
    Add-Type -AssemblyName PresentationFramework
    [System.Windows.MessageBox]::Show('Demo 启动未完成。请检查文件夹是否完整，以及 Python 运行环境是否可用。','BuildReasonSeg Demo') | Out-Null
    exit 1
}
