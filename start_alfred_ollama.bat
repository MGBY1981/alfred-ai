@echo off
rem ---------------------------------------------------------------
rem Alfred, the full local route - for trying it out, without
rem changing config.py.
rem
rem These two variables apply to THIS WINDOW ONLY. Nothing is written
rem anywhere, start_alfred.bat is unaffected, and closing the window
rem undoes it. Set the values in config.py if you want them to stick.
rem
rem Needs a running Ollama with both models:
rem     ollama pull llama3.1:8b        extraction
rem     ollama pull nomic-embed-text   storage
rem
rem Remove one of the two lines to go half local - for instance keep
rem the cloud filter and only extract locally. The sidebar of the
rem running chat shows which stage ended up where.
rem ---------------------------------------------------------------

set ALFRED_EXTRACTOR_BACKEND=ollama
set ALFRED_FILTER_MODE=ollama

python -m streamlit run .\alfred_ollama.py --server.port=8505 --server.address=localhost
