# Function to read a JSONL file into a data frame
read_jsonl <- function(file_path) {
  lines <- readLines(file_path)
  parsed_list <- lapply(lines, function(line) {
    parsed <- fromJSON(line)
    if (is.list(parsed) && length(parsed) == 1) {
      return(parsed[[1]])
    } else {
      return(parsed)
    }
  })
  df <- do.call(rbind, lapply(parsed_list, function(x) as.data.frame(x, stringsAsFactors = FALSE)))
  return(df)
}

# This method aids in reading list columns
read_jsonl_stream <- function(file_path) {
  con <- file(file_path, open = "r", encoding = "UTF-8")
  on.exit(close(con))
  
  # stream_in reads JSON objects line-by-line
  data <- jsonlite::stream_in(con, verbose = FALSE)
  return(data)
}

# Add Doc ID to Known Docs
create_temp_doc_id <- function(input_text) {
  match <- regexec("\\[(.*?)\\]", input_text, perl = TRUE)
  match_result <- regmatches(input_text, match)
  
  if (length(match_result[[1]]) > 1) {
    extracted_text <- match_result[[1]][2]
    cleaned_text <- gsub("[^\\w]", "_", extracted_text, perl = TRUE)
    final_text <- gsub("_{2,}", "_", cleaned_text, perl = TRUE)
    return(tolower(final_text))
  }
  return(NULL)
}