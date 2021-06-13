(
  "%PREFIX%\Scripts\jupyter-contrib-nbextension.exe" uninstall --sys-prefix
  if errorlevel 1 exit 1
) >>"%PREFIX%\.messages.txt" 2>&1