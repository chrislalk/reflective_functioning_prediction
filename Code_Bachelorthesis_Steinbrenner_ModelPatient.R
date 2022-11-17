## 500 woerter pro dokument

# packages
library(quanteda)
library(readtext)
library(stringr)
library(topicmodels)
library(stm)
library(LDAvis)
library(udpipe)
library(dplyr)
# library("geometry") # für stm
# library(rsvd) # für stm
# library(Rtsne) # für stm
ud_model <- udpipe_download_model(language = "german")

setwd("C:/Users/jlu-su/Desktop/Tobias/patient")


# read transcripts
my_texts2 <- readtext::readtext("C:/Users/jlu-su/Desktop/Tobias/patient/Patient_Transkripte", encoding = "LATIN1")


# clean data. Important: Don't delete punctuation marks (important for lemmatization)
my_texts2$text <- gsub("/", " ", my_texts2$text)
my_texts2$text  <- gsub("_", " ", my_texts2$text)
my_texts2$text  <- gsub("-", " ", my_texts2$text)
#my_texts$text  <- gsub(",", " ", my_texts$text)
my_texts2$text  <- gsub("'", " ", my_texts2$text)
my_texts2$text  <- gsub("\\(.+?\\)", " ", my_texts2$text) #remove bracketed items
my_texts2$text  <- gsub('[0-9]*','',my_texts2$text)
my_texts2$text  <- gsub("T:", " ", my_texts2$text)
my_texts2$text  <- gsub("P:", " ", my_texts2$text)



# Add Metadata to dataframe
## No 
no <- 1:length(my_texts2$doc_id)

## SPEAKER (patient or therapeut)
speaker <- ifelse(str_detect(my_texts2$doc_id, pattern = "pat*"), "patient", "therapeut")

## Define Variable Patientcode
library(stringr)
pat_code <- str_sub(my_texts2$doc_id, -14, -8)

## Add session number
session_no <- str_sub(my_texts2$doc_id, -6, -5)
session_no <- as.numeric(session_no)

## add variables to data frame
z2 <- cbind(no, my_texts2, speaker, pat_code, session_no)

## Add symptom severity df
hscl_table <- read.csv ("C:/Users/jlu-su/Desktop/Tobias/therapeut/HSCL-11/topic_models_hscl.csv", header = TRUE, stringsAsFactors = FALSE)
hscl_table <- data.frame(1:3396,hscl_table$CODE, hscl_table$sitzung,hscl_table$Gesamtscore_hscl)
na.omit(hscl_table$hscl_table.sitzung)
complete.cases(hscl_table)
hscl_table <- hscl_table[complete.cases(hscl_table), ]

df4 <- data.frame(no=(character()),
                  symptom_sev=character(),
                  symptom_sev_bin=character(),
                  stringsAsFactors=FALSE) 



for (i in 1:nrow(z2)) {
  print(i)
  for (j in 1:nrow(hscl_table)) {
    if (z2[i, 5] == hscl_table[j,2] & z2[i,6] == hscl_table[j,3]) {
      df4[i,2] <- hscl_table[j,4]
    }
  }
}

# die Zeilen 80-119 kann bei Überprüfung des Codes ausgelassen werden, weil ich in einen Ordner die Transkripte abgelegt habe, die bei der Analyse berücksichtigt wurden

## Count NAs for Symptom Severity
df5 <- na.omit(df4$symptom_sev)
### -> 9 NAs, insgesamt also 447 Daten



## convert column "session_no" to numeric
df4$symptom_sev <- as.numeric(df4$symptom_sev )



df7<- data.frame(bin=(character()),
                 stringsAsFactors=FALSE) 

## transform symptom severity into binary
for (i in 1:nrow(df4)) {
  if (is.na(df4$symptom_sev[i])) {
    df7[i,1] <- NA
  }
  else if (df4$symptom_sev[i] >=2.09){
    df7[i,1] <-1
  }
  else if (df4$symptom_sev[i] <2.09) {
    df7[i,1] <- 0
  }
}
z2$symptom_sev <- df4$symptom_sev
z2$symptom_sev_bin <- df7$bin

# neun Zeilen mit NAs löschen. Diese Transkripte können nicht berücksichtigt werden, da STM bei estimateEffect keine NAs haendeln kann
## 149
## 181
## 192
## 218
## 287
## 291
## 420
## 432
## 441
z2 <- z2[-c(149, 181, 192, 218, 287, 291, 420, 432, 441),]
#z2 <- z2$doc_id[-c("patient.1405P09_85.txt","patient.1405P09_65.txt"),]

## hier herausgefunden, welche Skripte ich aussortiert habe (siehe Wordtable)
#x <- data.frame(no=(character()))
#y <- data.frame(no=(factor()))



#divide documents into single speech turns
tt2 <- str_split(z2$text, pattern = "\n", simplify = T)



# connect single speech turns and count the number of words (500 per document. An dieser Stelle wird es dann für die Dokumentenlänge auf 100 bzw.1000 Wörter angepasst. Oder auch weggelassen für ganzes Transkript als Dokument)
start.time10 <- Sys.time()
for (a in 1:nrow(tt2)) {
  print(a)
  for (b in 1:ncol(tt2)) {
    for (c in b+1:ncol(tt2)) {
      if (str_count(tt2[ a, b], pattern = '\\w+') < 500  & b+c <= ncol(tt2)) {
                tt2[ a,b] <- paste(tt2[ a,b] , tt2[ a,b+c] , sep = " ")
        tt2[ a, b+c] <- ""
      }
      else
        break
    }
  }
}
end.time10 <- Sys.time()
time.taken10 <- end.time10 - start.time10
time.taken10
##time taken around 9 hours.
save(list = c("tt2"), file = "stm500_Patient_symptom_sev.Rdata")


# create new data frame. necessary to connect text (documents) with metadata again.
df2 <- data.frame(no=(character()),
                  doc_id=character(), 
                  speaker=character(), 
                  pat_code=character(), 
                  session_no=character(),
                  short_text=character(),
                  symptom_sev=character(),
                  symptom_sev_bin=character(),
                  stringsAsFactors=FALSE) 

v2 <- data.frame(no=(character()),
                 doc_id=character(), 
                 speaker=character(), 
                 pat_code=character(), 
                 session_no=character(),
                 short_text=character(),
                 symptom_sev=character(),
                 symptom_sev_bin=character(),
                 stringsAsFactors=FALSE) 

# connect text (documents) with metadata.
for (a in 1:nrow(tt2)) {
  for(b in 1:ncol(tt2)) {
    v2[b ,6] <- tt2[ a,b]
    v2[b, 1] <- z2[a, 1]
    v2[b, 2] <- z2[a, 2]
    v2[b, 3] <- z2[a, 4]
    v2[b, 4] <- z2[a, 5]
    v2[b, 5] <- z2[a, 6]
    v2[b, 7] <- z2[a, 7]
    v2[b, 8] <- z2[a, 8]
  }
  df2 <- rbind(df2, v2)
}



# Delete empty rows. 
test20 <- df2[str_count(df2$short_text, pattern = '\\w+') >= 500, ]


## Lemmatization with udpipe
ud_model_german <- udpipe_load_model("C:/Users/jlu-su/Desktop/Tobias/therapeut/Udpipe Model/german-gsd-ud-2.5-191206.udpipe")


start.time2 <- Sys.time()
x62 <- udpipe_annotate(ud_model_german, x = iconv(test20$short_text), tagger = "default", parser = "none", trace = T)
#x62 <- udpipe_annotate(ud_model_german, x = iconv(test20$short_text, to = "UTF-8"), tagger = "default", parser = "none", trace = T)
x72 <- as.data.frame(x62)
end.time2 <- Sys.time()
time.taken2 <- end.time2 - start.time2
time.taken2


## connect columns
x72$doc_id <- as.factor(x72$doc_id)
x72$sentence <- NULL
text_lemma2 <- c()
for(id in unique(x72$doc_id)) {
  subset_paste2 = subset(x72, doc_id == id)
  string_lemma2 <- paste(subset_paste2$lemma, collapse = ' ')
  text_lemma2 <- c(text_lemma2, string_lemma2)
} 

## append lemma to data frame
paste_test2 <- paste(x72$lemma, collapse=' ')
test20$text_lemma <- text_lemma2

## convert column "session_no" to numeric
test20$session_no <- as.numeric(test20$session_no)
sapply(test20, class)
test20$symptom_sev <- as.numeric(test20$symptom_sev)





# create corpus
w2 <- corpus(test20, unique_docnames= F, text_field = "text_lemma")

# Create Stopword_List
own_stopwords <- readLines("C:/Users/jlu-su/Desktop/Tobias/stopwords/stopwords.txt", encoding = "UTF-8")
tobi_stopwords <-readLines("C:/Users/jlu-su/Desktop/Tobias/stopwords/tobi_stopwords.txt", encoding = "UTF-8")
april_stopwords <-readLines("C:/Users/jlu-su/Desktop/Tobias/stopwords/stopwords27.04..txt", encoding = "UTF-8")



# Tokenization + remove numbers + remove punctuation + remove separators + remove Stopwords. Ggf stopwords und/oder  Trimming weglassen, damit man sieht wie viel und was gefiltert wird. 
tokens_x20 <- tokens(w2,remove_punct = T, remove_symbols = T, remove_numbers = T, remove_separators = T, include_docvars = T) %>%
  tokens_tolower() %>%
  tokens_remove(pattern = own_stopwords) %>%
  tokens_remove(pattern = tobi_stopwords) %>%
  tokens_remove(pattern = april_stopwords) %>%
  tokens_remove(pattern = ".." ) %>%
  tokens_remove(pattern = "...") 

## Let's create a document-feature-matrix.
### -> min term frequency = 5
dfm_tok_meta2 <- quanteda::dfm(tokens_x20, include_docvars = T) %>%
  dfm_trim(min_termfreq = 5) 

tok_dfm2stm2 <- quanteda::convert(dfm_tok_meta2, to = "stm")

save(list = c("hscl_table","tt2", "test20", "x72", "dfm_tok_meta2", "tok_dfm2stm2"), file = "stm500_Patient_symptom_sev.Rdata")



## What is a good number of Topics?
### Calculate Spectral Initialization
## attention: DFM is modified without stopwords
meta_stm_p_symp <- stm(documents =tok_dfm2stm2$documents, vocab = tok_dfm2stm2$vocab, K = 0,prevalence = ~symptom_sev_bin, max.em.its = 500, data = tok_dfm2stm2$meta, init.type = "Spectral")
#### -> 88 Topics
save(list = c("tt2", "test20", "x72", "dfm_tok_meta2", "tok_dfm2stm2", "meta_stm_p_symp" ), file = "stm500_Patient.Rdata")



findingk <- searchK(tok_dfm2stm2$documents, tok_dfm2stm2$vocab, K = c(60,70,80,90,100,110,120), data = tok_dfm2stm2$meta, init.type = "Spectral")
save("findingk", file = "findingk_patient_500")
plot(findingk)



### calculate 3 topic models and rate interpretability scores
meta_stm_p_symp_70 <- stm(documents =tok_dfm2stm2$documents, vocab = tok_dfm2stm2$vocab, K = 70,prevalence = ~symptom_sev_bin, max.em.its = 500, data = tok_dfm2stm2$meta, init.type = "Spectral")
meta_stm_p_symp_80 <- stm(documents =tok_dfm2stm2$documents, vocab = tok_dfm2stm2$vocab, K = 80,prevalence = ~symptom_sev_bin, max.em.its = 500, data = tok_dfm2stm2$meta, init.type = "Spectral")
meta_stm_p_symp_100 <- stm(documents =tok_dfm2stm2$documents, vocab = tok_dfm2stm2$vocab, K = 100,prevalence = ~symptom_sev_bin, max.em.its = 500, data = tok_dfm2stm2$meta, init.type = "Spectral")

save(list = c("meta_stm_p_symp_70","meta_stm_p_symp_80","meta_stm_p_symp_100"), file = "labelTopics_patient_70_vs_80_vs_100.Rdata")


# Top 10 most salient terms were also inspected via LDAvis
toLDAvis(meta_stm_p_symp_80, tok_dfm2stm2$documents)

### Label associated words within a topic for human validation
labelTopics(meta_stm_p_symp_70)
labelTopics(meta_stm_p_symp_80)
labelTopics(meta_stm_p_symp_100)




### Lets examine documents that are highly associated with topics
thoughts76 <- findThoughts(meta_stm_p_symp_80, texts = test20$short_text, n = 2, topic = 76)
thoughts76 <- findThoughts(meta_stm_p_symp_80, texts = test20$short_text, n = 1000, topic = 76)



## top ten most frequent topics above all documents: Were inspected via LDAvis. Da Nummern in LDAvis unterschiedlich, Zuordnung mit LabelTopics per Hand
toLDAvis(meta_stm_p_symp_80,tok_dfm2stm2$documents)
plot(meta_stm_p_symp_80, type ="summary", topics: c(23,76,66,31,2,24,14,71,61,53))










## metadata relationship
prep <- estimateEffect(1:80 ~ symptom_sev_bin, meta_stm_p_symp_80, meta = tok_dfm2stm2$meta, uncertainty = "Global")
prep
plot(prep, covariate = "symptom_sev_bin", topics = 1:80, model = meta_stm_p_symp_80, method = "difference", cov.value1 = "1", cov.value2 = "0", xlab = "low symptom severity ... high symptom severity", main = "Covariate high vs low symptom severity", xlim = c(-0.1, 0.1),labeltype = "custom", custom.labels = 1:80)
### hier prep-tabelle für t-werte und p ausgeben lassen. Dann signifikante Beziehungen zu Symptom Severity anschauen und nur diese als Plot ausgeben lassen
## for MP significant: 5,7,15,16,24,25,26,28,30,31,36,41,43,46,48,50,54,58,61,65,67,68,69,70,71,77
labels_mp <- c("5","basic school, play music (7)","15","have injection (16)","fear in the dark (24)","activities in village (25)","grandparents, family (26)","home time, holiday (28)","30","anger (31)",
               "death, funeral (36)","be upset, stress (41)","read (43)","studies (46)","drink alcohol (48)","50","psychotherapy (54)","58","61","65",
               "cognitions (67)","surgery (68)","69","70","doctor (71)","dole (77)")

plot.estimateEffect(prep, covariate = "symptom_sev_bin", topics = c(5,7,15,16,24,25,26,28,30,31,36,41,43,46,48,50,54,58,61,65,67,68,69,70,71,77), model = meta_stm_p_symp_80, method = "difference", cov.value1 = "1", cov.value2 = "0", xlab = "low symptom severity ... high symptom severity", main = "Covariate high vs low symptom severity", xlim = c(-0.05, 0.05),labeltype = "custom", custom.labels = labels_mp)




### Lets calculate quartils of the document length (Original Text)
quantile(str_count(test20$short_text,  pattern = '\\w+'))

long_text <- test20[str_count(test20$short_text, pattern = '\\w+') >= 1000, ]

### Lets calculate quartils of the document length (Lemmas in DFM and Topic Model)
quantile(ntoken(dfm_tok_meta2))



