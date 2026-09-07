$ErrorActionPreference = "Stop"
$dest = "results\musiccaps_original"
New-Item -ItemType Directory -Force -Path $dest | Out-Null
foreach ($name in @("bert","gnn","cnn","concat","fusion")) {
    $src = Join-Path "results" $name; $dst = Join-Path $dest $name
    if ((Test-Path $src) -and -not (Test-Path $dst)) { Copy-Item $src $dst -Recurse; Write-Host "Copied $src -> $dst" }
}
if (Test-Path "results\ablation_comparison.csv") { Copy-Item "results\ablation_comparison.csv" "$dest\ablation_comparison.csv" -Force }
Write-Host "Original MusicCaps results preserved under $dest"
