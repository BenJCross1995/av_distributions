@echo off
:: ===== Configurable Variables =====
set "corpus=Enron"
set "data_type=training"
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
set "save_loc=%nas_base_loc%/paraphrase examples/%corpus%"
set "script_loc=C:/Users/benjc/Documents/GitHub/av_distributions/scripts/run_openai_paraphrase_method.py"
set "credentials_loc=C:/Users/benjc/Documents/GitHub/av_distributions/credentials.json"
set "prompt_loc=C:/Users/benjc/Documents/GitHub/av_distributions/prompts/exhaustive_constrained_ngram_paraphraser_prompt_JSON.txt"

:: ===== Setup Project and Environment =====
cd /d "C:\Users\benjc\Documents\GitHub\av_distributions"
call venv\Scripts\activate

:: ====== Test Cases ======
call :run_test test_01 andy_zipper_mail_1 andy_zipper_mail_2	
call :run_test test_02 cara_semperger_mail_1 cara_semperger_mail_4
call :run_test test_03 barry_tycholiz_mail_3 barry_tycholiz_mail_2
call :run_test test_04 barry_tycholiz_mail_5 barry_tycholiz_mail_2
call :run_test test_05 d_thomas_mail_2 d_thomas_mail_3	
call :run_test test_06 daren_farmer_mail_5 daren_farmer_mail_3
call :run_test test_07 daren_farmer_mail_2 darrell_schoolcraft_mail_3
call :run_test test_08 k_allen_mail_3 kam_keiser_mail_4
call :run_test test_01 elizabeth_sager_mail_4 errol_mclaughlin_mail_3	
call :run_test test_09 cara_semperger_mail_1 cara_semperger_mail_4
call :run_test test_10 darrell_schoolcraft_mail_1 darron_giron_mail_1
call :run_test test_11 carol_clair_mail_4 chris_dorland_mail_3
call :run_test test_12 dan_hyvl_mail_4 dana_davis_mail_1

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
  --corpus "%corpus%" ^
  --data_type "%data_type%" ^
  --known_doc "%known_doc%" ^
  --unknown_doc "%unknown_doc%" ^
  --openai_model "%openai_model%" ^
  --max_tokens %max_tokens% ^
  --temperature %temperature% ^
  --n %n_samples% ^
  --credentials_loc %credentials_loc% ^
  --prompt_loc %prompt_loc%
  

endlocal
exit /b
