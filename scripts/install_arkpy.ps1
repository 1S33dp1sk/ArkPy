# scripts/install_arkpy.ps1
[CmdletBinding()]
param(
  [string]$Python = "",
  [string]$SourceDir = "",
  [ValidateSet('Auto','User','Machine')] [string]$PathScope = 'Auto',
  [switch]$Editable
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Resolve-Python([string]$Preferred){
  if ($Preferred) {
    try {
      & "$Preferred" -c "import sys;print('.'.join(map(str,sys.version_info[:3])))" | Out-Null
      return (Resolve-Path "$Preferred").Path
    } catch {}
  }
  $cands = @(
    @{exe='py'; args=@('-3')},
    @{exe='py'; args=@()},
    @{exe='python3'; args=@()},
    @{exe='python'; args=@()}
  )
  foreach ($c in $cands) {
    try {
      & $c.exe @($c.args) -c "import sys;print('.'.join(map(str,sys.version_info[:3])))" | Out-Null
      $p = & $c.exe @($c.args) -c "import sys,os;print(os.path.abspath(sys.executable))"
      if ($p) { return $p.Trim() }
    } catch {}
  }
  throw "Python 3.9+ not found. Install Python or pass -Python."
}

function Require-Python39Plus([string]$py){
  $v = & "$py" -c "import sys;print('.'.join(map(str,sys.version_info[:3])))"
  $ver = [Version]$v
  if ($ver.Major -lt 3 -or ($ver.Major -eq 3 -and $ver.Minor -lt 9)) {
    throw "Python >= 3.9 required, found $v at $py"
  }
}

function Ensure-Pip([string]$py){
  try { & "$py" -m pip --version | Out-Null } catch { & "$py" -m ensurepip --upgrade | Out-Null }
  & "$py" -m pip install --upgrade pip setuptools wheel | Out-Null
}

function Find-ProjectDir([string]$hint){
  $cands = @()
  if ($hint)        { $cands += (Resolve-Path $hint).Path }
  if ($PSScriptRoot){ $cands += $PSScriptRoot }
  $cands += (Get-Location).Path
  foreach($d in $cands){
    if (Test-Path (Join-Path $d 'pyproject.toml')) { return $d }
  }
  throw "pyproject.toml not found. Pass -SourceDir pointing to ArkPy root."
}

function Scripts-Dir([string]$py){
  (& "$py" -c "import sysconfig;print(sysconfig.get_paths()['scripts'])").Trim()
}

function Test-Admin {
  $wi = [Security.Principal.WindowsIdentity]::GetCurrent()
  $wp = New-Object Security.Principal.WindowsPrincipal($wi)
  return $wp.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Add-ToPath([string]$dir, [string]$scope){
  if (-not (Test-Path $dir)) { throw "Scripts dir not found: $dir" }
  if ($scope -eq 'Machine') { $sc = 'Machine' } else { $sc = 'User' }
  $current = [Environment]::GetEnvironmentVariable('Path', $sc)
  if (-not $current) { $current = '' }
  $parts = @()
  foreach($p in $current.Split(';')){ if($p -and $p.Trim() -ne ''){ $parts += $p.Trim() } }
  $present = $false
  foreach($p in $parts){ if ($p.TrimEnd('\') -ieq $dir.TrimEnd('\')) { $present = $true; break } }
  if (-not $present){
    $newPath = ($parts + $dir) -join ';'
    [Environment]::SetEnvironmentVariable('Path', $newPath, $sc)
  }
  $sessParts = $env:Path.Split(';')
  $sessPresent = $false
  foreach($p in $sessParts){ if ($p.TrimEnd('\') -ieq $dir.TrimEnd('\')) { $sessPresent = $true; break } }
  if (-not $sessPresent){ $env:Path = "$($env:Path);$dir" }
}

# resolve python
$py = Resolve-Python -Preferred $Python
Require-Python39Plus $py
Ensure-Pip $py

# locate project
$proj = Find-ProjectDir -hint $SourceDir

# install (no CLI execution)
if ($Editable) {
  & "$py" -m pip install --upgrade -e "$proj"
} else {
  & "$py" -m pip install --upgrade "$proj"
}

# PATH setup
$scripts = Scripts-Dir $py
if ($PathScope -eq 'Auto') { if (Test-Admin) { $PathScope = 'Machine' } else { $PathScope = 'User' } }
Add-ToPath -dir $scripts -scope $PathScope

# verify import and shim presence
$ver = (& "$py" -c "import importlib; m=importlib.import_module('arknet_py'); import sys; print(getattr(m,'__version__','unknown'))" 2>$null).Trim()
if (-not $ver) { $ver = "unknown" }

$shimPresent = $false
foreach($n in @('arknet-py.exe','arknet-py','arknet-py-script.py')){
  if (Test-Path (Join-Path $scripts $n)) { $shimPresent = $true; break }
}
$inPath = $false
if (Get-Command arknet-py -ErrorAction SilentlyContinue) { $inPath = $true }

Write-Host ("arknet-py installed. version: {0}" -f $ver)
Write-Host ("scripts dir: {0}" -f $scripts)
Write-Host ("shim present: {0}" -f ($(if($shimPresent){"yes"}else{"no"})))
Write-Host ("on PATH now: {0}" -f ($(if($inPath){"yes"}else{"no"})))
if (-not $inPath) { Write-Host "open a new terminal to refresh PATH." }
