@echo off
setlocal EnableExtensions

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..\..") do set "ROOT=%%~fI"

for /f %%I in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd-HHmmss"') do set "RUN_ID=%%I"

if not defined HGT_CONDA_ENV set "HGT_CONDA_ENV=HGT"
if not defined HGT_DATA_DIR set "HGT_DATA_DIR=%ROOT%\data"
if not defined HGT_OUTPUT_DIR set "HGT_OUTPUT_DIR=%ROOT%\outputs\preprocess\hgt_graph\runs\%RUN_ID%"

set "EXTRA_ARGS="
if defined HGT_STATIONS_CSV set EXTRA_ARGS=%EXTRA_ARGS% --stations-csv "%HGT_STATIONS_CSV%"
if defined HGT_BUILDINGS_CSV set EXTRA_ARGS=%EXTRA_ARGS% --buildings-csv "%HGT_BUILDINGS_CSV%"
if defined HGT_CONSERVATION_GEOJSON set EXTRA_ARGS=%EXTRA_ARGS% --conservation-geojson "%HGT_CONSERVATION_GEOJSON%"
if defined HGT_ADMIN_GEOJSON set EXTRA_ARGS=%EXTRA_ARGS% --admin-geojson "%HGT_ADMIN_GEOJSON%"
if defined HGT_ROADS_CSV set EXTRA_ARGS=%EXTRA_ARGS% --roads-csv "%HGT_ROADS_CSV%"
if defined HGT_POI_CSV set EXTRA_ARGS=%EXTRA_ARGS% --poi-csv "%HGT_POI_CSV%"

echo [build_hgt_graph] env=%HGT_CONDA_ENV%
echo [build_hgt_graph] data-dir=%HGT_DATA_DIR%
echo [build_hgt_graph] output-dir=%HGT_OUTPUT_DIR%

conda run -n "%HGT_CONDA_ENV%" python "%SCRIPT_DIR%build_hgt_graph.py" ^
  --data-dir "%HGT_DATA_DIR%" ^
  --output-dir "%HGT_OUTPUT_DIR%" ^
  %EXTRA_ARGS% %*

if errorlevel 1 (
  echo [build_hgt_graph] failed
  exit /b %errorlevel%
)

echo [build_hgt_graph] completed: %HGT_OUTPUT_DIR%
endlocal
