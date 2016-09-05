library(dplyr)
library(syuzhet)
library(tm)
library(RWeka)
library(SnowballC)  
library(stringr)

prepareString <- function (x) {
  x <- gsub("on't", 'ont', x)
  x <- gsub("sn't", 'oesnt', x)
  x <- gsub("'ve", 've', x)
  x <- iconv(x, 'latin1', 'ASCII', sub='')
  x <- tolower(x)
  x <- removePunctuation(x)
  x <- removeNumbers(x)
  x <- stripWhitespace(x)
  str_trim(x)
}

buildDTMatrix <- function (corpus, min, max, language = 'english') {
  DocumentTermMatrix(
    corpus, 
    control = list(
      tokenize = function(x) {
        NGramTokenizer(x, Weka_control(min = min, max = max))
      },
      language = language,
      stemWords = FALSE
    )
  )
}

setwd('~/experiments/freeform')

topics    <- read.csv('topics.csv', stringsAsFactors = FALSE)
questions <- read.csv('questions.csv', stringsAsFactors = FALSE) 
answers   <- read.csv('answers.csv', stringsAsFactors = FALSE)
  
head(topics)
head(questions)
head(answers)

question.text <- function (id) { questions[questions$id == id,]$body }

answers <- subset(answers, company_id == 3)
answers$question <- sapply(answers$poll_id, question.text)
answers$answer_sentiment <- get_nrc_sentiment(answers$text)
answers$question_sentiment <- get_nrc_sentiment(answers$question)

poll_87 <- subset(answers, poll_id==87)
poll_32 <- subset(answers, poll_id==32)

# sentiments
barplot(sort(colSums(poll_87$answer_sentiment)), horiz = TRUE, cex.names = 0.7, las = 1, main= questions[questions$id == 87,]$body)

# word frequencies

corpus <- Corpus(VectorSource(paste(poll_87$text, collapse = ' ')))
corpus <- tm_map(corpus, prepareString)
corpus <- tm_map(corpus, PlainTextDocument)
unigram_sparse_dtm <- buildDTMatrix(corpus, 1, 1)
matrix <- as.matrix(unigram_sparse_dtm)
uni_sorted <- matrix[1,order(matrix[1,])]
barplot(tail(uni_sorted, n =10), horiz = TRUE, cex.names = 0.7, las = 1, main= questions[questions$id == 87,]$body)

corpus <- Corpus(VectorSource(paste(poll_32$text, collapse = ' ')))
corpus <- tm_map(corpus, prepareString)
corpus <- tm_map(corpus, PlainTextDocument)
unigram_sparse_dtm <- buildDTMatrix(corpus, 1, 1)
matrix <- as.matrix(unigram_sparse_dtm)
uni_sorted <- matrix[1,order(matrix[1,])]
barplot(tail(uni_sorted, n =10), horiz = TRUE, cex.names = 0.7, las = 1, main= questions[questions$id == 32,]$body)