@echo off
:: ===== Configurable Variables =====
set "corpus=Wiki"
set "data_type=test"
set "nas_base_loc=//bc_nas_storage/BCross"
set "model_name=Qwen2.5-0.5B-Instruct"
set "openai_model=gpt-4.1"
set "max_tokens=5000"
set "temperature=0.7"
set "n_samples=10"

:: ===== Derived Paths =====
set "known_loc=%nas_base_loc%/datasets/author_verification/%data_type%/%corpus%/known_raw.jsonl"
set "unknown_loc=%nas_base_loc%/datasets/author_verification/%data_type%/%corpus%/unknown_raw.jsonl"
set "metadata_loc=%nas_base_loc%/datasets/author_verification/%data_type%/metadata.rds"
set "model_loc=%nas_base_loc%/models/Qwen 2.5/%model_name%"
set "save_loc=%nas_base_loc%/paraphrase examples/%corpus%-%data_type%-gpt-4-1"
set "completed_loc=%nas_base_loc%/paraphrase examples/%corpus%-%data_type%-gpt-4-1-completed"
set "script_loc=C:/Users/benjc/Documents/GitHub/av_distributions/scripts/run_openai_paraphrase_method.py"
set "credentials_loc=C:/Users/benjc/Documents/GitHub/av_distributions/credentials.json"
set "prompt_loc=C:/Users/benjc/Documents/GitHub/av_distributions/prompts/exhaustive_constrained_ngram_paraphraser_prompt_JSON_newv2.txt"

:: ===== Setup Project and Environment =====
cd /d "C:\Users\benjc\Documents\GitHub\av_distributions"
call venv\Scripts\activate

:: ====== Test Cases ======
call :run_test test_01 ivoshandor_text_4 jasper_deng_text_4
call :run_test test_02 ivoshandor_text_5 jasper_deng_text_4
call :run_test test_03 ivoshandor_text_2 jasper_deng_text_4
call :run_test test_04 mathsci_text_3 maunus_text_1
call :run_test test_05 mathsci_text_2 maunus_text_1
call :run_test test_06 mathsci_text_11 maunus_text_1
call :run_test test_07 u21980_text_1 updown_text_3
call :run_test test_08 u21980_text_4 updown_text_3
call :run_test test_09 u21980_text_3 updown_text_3
call :run_test test_10 jbmurray_text_5 jc37_text_4
call :run_test test_11 jbmurray_text_4 jc37_text_4
call :run_test test_12 jbmurray_text_2 jc37_text_4

echo All tests complete.
pause
exit /b


:: ====== Function to Run One Test ======
:run_test
setlocal
set "test_name=%1"
set "known_doc=%2"
set "unknown_doc=%3"

echo Running %test_name%: %known_doc% vs %unknown_doc%

python -u "%script_loc%" ^
  --known_loc "%known_loc%" ^
  --unknown_loc "%unknown_loc%" ^
  --metadata_loc "%metadata_loc%" ^
  --model_loc "%model_loc%" ^
  --save_loc "%save_loc%" ^
  --completed_loc "%completed_loc%" ^
  --corpus "%corpus%" ^
  --data_type "%data_type%" ^
  --known_doc "%known_doc%" ^
  --unknown_doc "%unknown_doc%" ^
  --openai_model "%openai_model%" ^
  --max_tokens %max_tokens% ^
  --temperature %temperature% ^
  --n %n_samples% ^
  --credentials_loc %credentials_loc% ^
  --prompt_loc %prompt_loc% ^
  --score_texts
  

endlocal
exit /b
