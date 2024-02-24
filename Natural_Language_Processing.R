library(tidyverse)
library(data.table)
library(text2vec)
library(caTools)
library(glmnet)



e <- read.csv('emails.csv')
e %>% view()


e$ID <- seq.int(nrow(e))

e <- e %>% select(ID,everything())


lapply(e,class)
e$spam %>% table %>% prop.table()


set.seed=123
split <- e$spam %>% sample.split(SplitRatio = 0.8)
train <- e %>% subset(split==T)
test <- e %>% subset(split==F)

train %>% dim()

test %>% dim()

it_train <- train$text %>% 
  itoken(preprocessor = tolower, 
         tokenizer = word_tokenizer,
         ids = train$ID,
         progressbar = F) 

vocab <- it_train %>% create_vocabulary()

vocab %>% 
  arrange(desc(term_count)) %>% 
  head(110) %>% 
  tail(10) 

vectorizer <- vocab %>% vocab_vectorizer()
dtm_train <- it_train %>% create_dtm(vectorizer)

dtm_train %>% dim()



glmnet_classifier <- dtm_train %>% 
  cv.glmnet(y = train[['spam']],
            family = 'binomial', 
            type.measure = "auc",
            nfolds = 10,
            thresh = 0.001,
            maxit = 1000)

glmnet_classifier$cvm %>% max() %>% round(3) %>% paste("-> Max AUC")



it_test <- test$text %>% tolower() %>% word_tokenizer()

it_test <- it_test %>% 
  itoken(ids = test$ID,
         progressbar = F)


dtm_test <- it_test %>% create_dtm(vectorizer)

preds <- predict(glmnet_classifier, dtm_test, type = 'response')[,1]
glmnet:::auc(test$spam, preds) %>% round(2)


# Prune some words and check its effect on the result ----
stop_words <- c("i", "you", "he", "she", "it", "we", "they",
                "me", "him", "her", "them",
                "my", "your", "yours", "his", "our", "ours",
                "myself", "yourself", "himself", "herself", "ourselves",
                "the", "a", "an", "and", "or", "on", "by", "so",
                "from", "about", "to", "for", "of", 
                "that", "this", "is", "are")

vocab <- it_train %>% create_vocabulary(stopwords = stop_words)


pruned_vocab <- vocab %>% 
  prune_vocabulary(term_count_min = 10, 
                   doc_proportion_max = 0.5,
                   doc_proportion_min = 0.001)

pruned_vocab %>% 
  arrange(desc(term_count)) %>% 
  head(10) 

vectorizer <- pruned_vocab %>% vocab_vectorizer()

dtm_train <- it_train %>% create_dtm(vectorizer)
dtm_train %>% dim()

glmnet_classifier <- dtm_train %>% 
  cv.glmnet(y = train[['spam']], 
            family = 'binomial',
            type.measure = "auc",
            nfolds = 4,
            thresh = 0.001,
            maxit = 1000)

glmnet_classifier$cvm %>% max() %>% round(3) %>% paste("-> Max AUC")

dtm_test <- it_test %>% create_dtm(vectorizer)
preds <- predict(glmnet_classifier, dtm_test, type = 'response')[,1]
glmnet:::auc(test$spam, preds) %>% round(2)

# change on the result after ngrams

vocab <- it_train %>% create_vocabulary(ngram = c(1L, 3L))

vocab <- vocab %>% 
  prune_vocabulary(term_count_min = 10, 
                   doc_proportion_max = 0.5)

ngram_vectorizer <- vocab %>% vocab_vectorizer()

dtm_train <- it_train %>% create_dtm(ngram_vectorizer)
dtm_train %>% dim()

glmnet_classifier <- dtm_train %>% 
  cv.glmnet(y = train[['spam']], 
            family = 'binomial',
            type.measure = "auc",
            nfolds = 5,
            thresh = 0.001,
            maxit = 1000)

glmnet_classifier$cvm %>% max() %>% round(3) %>% paste("-> Max AUC")

dtm_test <- it_test %>% create_dtm(ngram_vectorizer)
preds <- predict(glmnet_classifier, dtm_test, type = 'response')[,1]
glmnet:::auc(test$spam, preds) %>% round(2)




