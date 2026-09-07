$ErrorActionPreference = "Stop"
$Root = (Resolve-Path ".").Path
$Raw = Join-Path $Root "data\raw"
$MetaZip = Join-Path $Raw "fma_metadata.zip"
$MediumZip = Join-Path $Raw "fma_medium.zip"
New-Item -ItemType Directory -Force -Path $Raw | Out-Null
Write-Host "Downloading FMA metadata (~342 MiB) to the project drive..."
curl.exe -L "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip" -o $MetaZip
Write-Host "Downloading FMA-medium (~22 GiB) to the project drive..."
curl.exe -L "https://os.unil.cloud.switch.ch/fma/fma_medium.zip" -o $MediumZip
Write-Host "Extracting metadata..."
Expand-Archive -Path $MetaZip -DestinationPath $Raw -Force
Write-Host "Extracting FMA-medium..."
Expand-Archive -Path $MediumZip -DestinationPath $Raw -Force
Write-Host "Expected: data\raw\fma_metadata\tracks.csv and data\raw\fma_medium\000\*.mp3"
Write-Host "After verifying extraction, delete the ZIP files to recover disk space."
