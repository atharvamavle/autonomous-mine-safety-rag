$ErrorActionPreference = "Stop"

New-Item -ItemType Directory -Force -Path "data\raw\manuals"  | Out-Null
New-Item -ItemType Directory -Force -Path "data\raw\incidents" | Out-Null

function Download-List($listPath, $outDir) {
  Get-Content $listPath | ForEach-Object {
    $url = $_.Trim()
    if ($url -eq "" -or $url.StartsWith("#")) { return }

    $fileName = Split-Path ($url.Split("?")[0]) -Leaf
    $outPath = Join-Path $outDir $fileName

    Write-Host "Downloading $fileName"
    Invoke-WebRequest -Uri $url -OutFile $outPath
  }
}

Download-List "scripts\manifests\safework_manuals.txt" "data\raw\manuals"
Download-List "scripts\manifests\rshq_incidents.txt"   "data\raw\incidents"
Download-List "scripts\manifests\nsw_bulletins.txt"    "data\raw\incidents"
