@echo off

@REM There is some ability to run 32-bit and 64-bit applications on Windows. If the
@REM default subdir is 32-bit, we do not want to show this message (as there is no oneDAL).
@REM So, if the conda subdir is not win-64 -> exit.
conda config --show subdir | %SYSTEMROOT%\System32\find.exe /I "win-64"
if errorlevel 1 exit 1

(
echo.
echo.
echo     Windows 64-bit packages of scikit-learn can be accelerated using scikit-learn-intelex.
echo     More details are available here: https://intel.github.io/scikit-learn-intelex
echo.
echo     For example:
echo.
echo         $ conda install scikit-learn-intelex
echo         $ python -m sklearnex my_application.py
echo.
) >> "%PREFIX%\.messages.txt"
