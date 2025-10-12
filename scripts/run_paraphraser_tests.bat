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
call :run_test test_01 salix_alba_text_1 sally_season_text_1	
call :run_test test_02 salix_alba_text_4 sally_season_text_1
call :run_test test_03 salix_alba_text_2 sally_season_text_1
call :run_test test_04 nableezy_text_12 nathan_text_10
call :run_test test_05 nableezy_text_1 nathan_text_10
call :run_test test_06 nableezy_text_11 nathan_text_10
call :run_test test_07 vsevolodkrolikov_text_11 wiki_guy_16_text_1
call :run_test test_08 vsevolodkrolikov_text_12 wiki_guy_16_text_1
call :run_test test_09 vsevolodkrolikov_text_13 wiki_guy_16_text_1	
call :run_test test_10 nocrowx_text_5 notpietru_text_2
call :run_test test_11 nocrowx_text_2 notpietru_text_2
call :run_test test_12 nocrowx_text_3 notpietru_text_2
call :run_test test_13 notpietru_text_4 obamafan70_text_5
call :run_test test_14 notpietru_text_1 obamafan70_text_5
call :run_test test_15 notpietru_text_5 obamafan70_text_5
call :run_test test_16 the_four_deuces_text_10 tijfo098_text_2
call :run_test test_17 the_four_deuces_text_1 tijfo098_text_2
call :run_test test_18 the_four_deuces_text_3 tijfo098_text_2
call :run_test test_19 petersymonds_text_4 peter_james_text_4
call :run_test test_20 petersymonds_text_1 peter_james_text_4
call :run_test test_21 petersymonds_text_2 peter_james_text_4
call :run_test test_22 pro_lick_text_5 protonk_text_13
call :run_test test_23 pro_lick_text_2 protonk_text_13
call :run_test test_24 pro_lick_text_3 protonk_text_13
call :run_test test_25 sally_season_text_12 scheinwerfermann_text_10
call :run_test test_26 sally_season_text_2 scheinwerfermann_text_10
call :run_test test_27 sally_season_text_11 scheinwerfermann_text_10
call :run_test test_28 ivoshandor_text_4 jasper_deng_text_4
call :run_test test_29 ivoshandor_text_5 jasper_deng_text_4
call :run_test test_30 ivoshandor_text_2 jasper_deng_text_4
call :run_test test_31 mathsci_text_3 maunus_text_1
call :run_test test_32 mathsci_text_2 maunus_text_1
call :run_test test_33 mathsci_text_11 maunus_text_1
call :run_test test_34 u21980_text_1 updown_text_3
call :run_test test_35 u21980_text_4 updown_text_3
call :run_test test_36 u21980_text_3 updown_text_3
call :run_test test_37 jbmurray_text_5 jc37_text_4
call :run_test test_38 jbmurray_text_4 jc37_text_4
call :run_test test_39 jbmurray_text_2 jc37_text_4

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
  --sore_texts
  

endlocal
exit /b
