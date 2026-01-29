@echo off
setlocal enabledelayedexpansion

REM === Путь к OpenVPN Connect ===
set OVPN="C:\Program Files\OpenVPN Connect\OpenVPNConnect.exe"

REM === Папка с ovpn-файлами ===
set OVPN_DIR=%~dp0ovpn

REM === Импорт всех ovpn-файлов (если уже импортированы — ошибок не будет) ===
for %%f in ("%OVPN_DIR%\*.ovpn") do (
    echo Импорт: %%f
    %OVPN% import --file "%%f" >nul 2>&1
)

REM === Получить список профилей ===
for /f "tokens=1" %%i in ('%OVPN% --list ^| findstr /R "vpngate"') do (
    set servers[!count!]=%%i
    set /a count+=1
)

echo Найдено профилей: %count%

if %count%==0 (
    echo ❌ Нет импортированных профилей. Проверь папку ovpn.
    exit /b
)

REM === Функция получения IP ===
:GET_IP
for /f "delims=" %%i in ('powershell -command "(Invoke-WebRequest -UseBasicParsing https://api.ipify.org).Content"') do set CURRENT_IP=%%i
goto :eof

REM === Получить исходный IP ===
call :GET_IP
echo Текущий IP: %CURRENT_IP%
set last_ip=%CURRENT_IP%

REM === Перебор серверов ===
:TRY_AGAIN
set /a index=%RANDOM% %% %count%
set server=!servers[%index%]!

echo.
echo Подключаемся к: !server!
%OVPN% --disconnect >nul 2>&1
%OVPN% --connect !server!
timeout /t 8 >nul

REM === Проверить новый IP ===
call :GET_IP
echo Новый IP: %CURRENT_IP%

if "%CURRENT_IP%"=="%last_ip%" (
    echo IP не изменился. Пробуем другой сервер...
    goto TRY_AGAIN
)

echo.
echo ✅ IP успешно изменён!
echo Старый IP: %last_ip%
echo Новый IP: %CURRENT_IP%
exit /b
