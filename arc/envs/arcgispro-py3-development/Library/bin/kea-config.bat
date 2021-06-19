@echo off

IF "%1"=="" (
   echo kea-config.bat [OPTIONS]
   echo Options:
   echo     [--prefix]
   echo     [--version]
   echo     [--libs]
   echo     [--cflags]
   echo     [--includes]
   EXIT /B 1
) ELSE (
:printValue
    if "%1" neq "" (
	    IF "%1"=="--prefix" echo D:/bld/kealib_1585883557023/_h_env/Library
	    IF "%1"=="--version" echo 1.4.13
	    IF "%1"=="--cflags" echo -ID:/bld/kealib_1585883557023/_h_env/Library/include
	    IF "%1"=="--libs" echo -LIBPATH:D:/bld/kealib_1585883557023/_h_env/Library/lib libkea.lib 
	    IF "%1"=="--includes" echo D:/bld/kealib_1585883557023/_h_env/Library/include
		shift
		goto :printValue
    )
	EXIT /B 0
)
