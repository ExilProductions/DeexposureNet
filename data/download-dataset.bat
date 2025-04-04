@echo off
setlocal

:: Set the download link for the zip file
set "URL=https://huggingface.co/datasets/mebix/images/blob/main/txt2img-images.zip"

:: Set the destination folder for the ZIP file
set "DEST_DIR=%cd%\dataset"  :: This will store the downloaded ZIP file

:: Set the temporary folder where the ZIP file will be extracted
set "TEMP_DIR=%cd%\temp_extract"

:: Create necessary directories
mkdir "%DEST_DIR%"
mkdir "%TEMP_DIR%"

:: Download the ZIP file
echo Downloading ZIP file...
curl -L -o "%DEST_DIR%\dataset.zip" "%URL%"

:: Check if download was successful
if %errorlevel% neq 0 (
    echo Download failed!
    exit /b 1
)

:: Extract the ZIP file
echo Unzipping file...
tar -xf "%DEST_DIR%\dataset.zip" -C "%TEMP_DIR%"

:: Rename the extracted folder to "original"
echo Renaming the extracted folder to "original"...
ren "%TEMP_DIR%\*" "original"

:: Move the "original" folder to the destination directory
move "%TEMP_DIR%\original" "%DEST_DIR%"

:: Clean up by deleting the temporary extraction folder
rd /s /q "%TEMP_DIR%"

echo Process completed successfully!

endlocal
pause
