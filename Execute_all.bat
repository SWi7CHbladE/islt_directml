@echo off




:: BatchGotAdmin
:-------------------------------------
REM  --> Check for permissions
    IF "%PROCESSOR_ARCHITECTURE%" EQU "amd64" (
>nul 2>&1 "%SYSTEMROOT%\SysWOW64\cacls.exe" "%SYSTEMROOT%\SysWOW64\config\system"
) ELSE (
>nul 2>&1 "%SYSTEMROOT%\system32\cacls.exe" "%SYSTEMROOT%\system32\config\system"
)

REM --> If error flag set, we do not have admin.
if '%errorlevel%' NEQ '0' (
    echo Requesting administrative privileges...
    goto UACPrompt
) else ( goto gotAdmin )

:UACPrompt
    echo Set UAC = CreateObject^("Shell.Application"^) > "%temp%\getadmin.vbs"
    set params= %*
    echo UAC.ShellExecute "cmd.exe", "/c ""%~s0"" %params:"=""%", "", "runas", 1 >> "%temp%\getadmin.vbs"

    "%temp%\getadmin.vbs"
    del "%temp%\getadmin.vbs"
    exit /B

:gotAdmin
    pushd "%CD%"
    CD /D "%~dp0"

conda activate slt_directml

python -m signjowy train configs/8head/sign_8head_8batch.yaml && python -m signjoey train configs/8head/sign_8head_16batch.yaml && python -m signjoey train configs/8head/sign_8head_32batch.yaml && python -m signjoey train configs/8head/sign_8head_64batch.yaml && python -m signjoey train configs/8head/sign_8head_128batch.yaml && python -m signjoey train config/8head/sign_8head_256batch.yaml

python -m signjowy train configs/16head/sign_16head_8batch.yaml && python -m signjoey train configs/16head/sign_16head_16batch.yaml && python -m signjoey train configs/16head/sign_16head_32batch.yaml && python -m signjoey train configs/16head/sign_16head_64batch.yaml && python -m signjoey train configs/16head/sign_16head_128batch.yaml && python -m signjoey train config/16head/sign_16head_256batch.yaml

python -m signjowy train configs/32head/sign_32head_8batch.yaml && python -m signjoey train configs/32head/sign_32head_16batch.yaml && python -m signjoey train configs/32head/sign_32head_32batch.yaml && python -m signjoey train configs/32head/sign_32head_64batch.yaml && python -m signjoey train configs/32head/sign_32head_128batch.yaml && python -m signjoey train config/32head/sign_32head_256batch.yaml

python -m signjowy train configs/64head/sign_64head_8batch.yaml && python -m signjoey train configs/64head/sign_64head_16batch.yaml && python -m signjoey train configs/64head/sign_64head_32batch.yaml && python -m signjoey train configs/64head/sign_64head_64batch.yaml && python -m signjoey train configs/64head/sign_64head_128batch.yaml && python -m signjoey train config/64head/sign_64head_256batch.yaml


exit