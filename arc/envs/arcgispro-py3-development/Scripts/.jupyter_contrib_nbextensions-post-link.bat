"%PREFIX%\Scripts\jupyter-contrib-nbextension.exe" install --sys-prefix
if errorlevel 1 exit 1
(
  "%PREFIX%\Scripts\jupyter-nbextension.exe" enable --sys-prefix collapsible_headings/main
  if errorlevel 1 exit 1
) >>"%PREFIX%\.messages.txt" 2>&1
